#include "Basic/Parallel.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

class ThreadPool {
public:
    ~ThreadPool() {
        StopWorkers();
    }

    int GetThreads() const {
        return threads_.load();
    }

    void SetThreads(int threads) {
        if (threads < 1)
            throw std::invalid_argument("threads must be at least 1");
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
        if (threads == GetThreads())
            return;
        StopWorkers();
        threads_.store(1);
        try {
            for (int index = 1; index < threads; ++index)
                workers_.emplace_back([this, index] { Worker(index); });
        } catch (...) {
            StopWorkers();
            throw;
        }
        threads_.store(threads);
    }

    void Run(std::size_t count, const std::function<void(std::size_t)>& task) {
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
        if (count == 0)
            return;
        if (count == 1 || workers_.empty()) {
            for (std::size_t index = 0; index < count; ++index)
                task(index);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &task;
            count_ = count;
            error_ = nullptr;
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        Execute(0, count, task);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return remaining_ == 0; });
        task_ = nullptr;
        if (error_)
            std::rethrow_exception(error_);
    }

private:
    void StopWorkers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
            worker.join();
        workers_.clear();
        stopping_ = false;
        generation_ = 0;
    }

    void Execute(std::size_t first, std::size_t count,
                 const std::function<void(std::size_t)>& task) {
        try {
            const std::size_t stride = GetThreads();
            for (std::size_t index = first; index < count; index += stride)
                task(index);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_)
                error_ = std::current_exception();
        }
    }

    void Worker(std::size_t index) {
        std::size_t generation = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            ready_.wait(lock, [this, generation] {
                return stopping_ || generation_ != generation;
            });
            if (stopping_)
                return;
            generation = generation_;
            const auto* task = task_;
            const std::size_t count = count_;
            lock.unlock();
            Execute(index, count, *task);
            lock.lock();
            if (--remaining_ == 0)
                finished_.notify_one();
        }
    }

    std::atomic<int> threads_{1};
    std::mutex dispatch_mutex_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::vector<std::thread> workers_;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::size_t count_ = 0;
    std::size_t remaining_ = 0;
    std::size_t generation_ = 0;
    std::exception_ptr error_;
    bool stopping_ = false;
};

ThreadPool& Pool() {
    static ThreadPool pool;
    return pool;
}

}

namespace Parallel {

void SetThreads(int threads) {
    Pool().SetThreads(threads);
}

int GetThreads() {
    return Pool().GetThreads();
}

void Run(std::size_t count, const std::function<void(std::size_t)>& task) {
    Pool().Run(count, task);
}

}
