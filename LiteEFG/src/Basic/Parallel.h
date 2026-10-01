#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <cstddef>
#include <functional>

namespace Parallel {

// The calling thread is included in this process-wide thread count.
void SetThreads(int threads);
int GetThreads();

// Execute each task exactly once and wait for the whole batch. Tasks must not
// invoke Run or SetThreads recursively. Any exception is rethrown on the caller.
void Run(std::size_t count, const std::function<void(std::size_t)>& task);

}

#endif
