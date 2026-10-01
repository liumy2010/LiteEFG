#include "Environment/Environment.h"

#include "Basic/Parallel.h"
#include "Computation/Static.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace {

bool IsRandom(const GraphNode& node) {
    return node.operation && node.operation->name.compare(0, 6, "Random") == 0;
}

// An inexpensive element-work estimate, not a prediction of elapsed time.
// Dynamic shapes use the action count when their size is not known statically.
std::vector<double> EstimateWork(Infoset& infoset, std::array<double, 2>& work) {
    std::vector<double> widths(infoset.results.size(), 1.0);
    for(size_t i=0; i<widths.size(); ++i)
        if(!infoset.results[i].empty()) widths[i] = infoset.results[i][0].size;
    for(const auto& node : infoset.graph.graph_nodes) {
        if(!node.operation) continue;
        const auto& name = node.operation->name;
        double width = 1.0, inputs = 0.0;
        for(int dependency : node.dependency) {
            width = std::max(width, widths[dependency]);
            inputs += widths[dependency];
        }
        if(name == "StaticConstVector" || IsRandom(node)) {
            if(node.dependency.empty())
                width = static_cast<StaticConstVector*>(node.operation.get())->elements.size;
            else {
                const auto& size = infoset.results[node.dependency[0]];
                width = (!size.empty() && size[0].size == 1) ?
                    std::max(0.0, size[0][0]) : double(infoset.children.size());
            }
        } else if(name == "Aggregate") width = node.dependency.size();
        else if(name == "Concat") width = inputs;
        else if(name == "Sum" || name == "Mean" || name == "Min" || name == "Max" ||
                name == "Dot" || name == "Euclidean" || name == "NegativeEntropy") width = 1.0;
        widths[node.idx] = width;
        if(node.status >= GraphNode::NodeStatus::backward_node) {
            double cost = 1.0 + inputs + width;
            if(name == "Projection") cost += width * std::log2(std::max(2.0, width));
            work[node.status - GraphNode::NodeStatus::backward_node] += cost;
        }
    }
    return widths;
}

}

void Environment::ResetParallelSchedule() {
    parallel_infosets.clear();
    parallel_layers.clear();
    for(auto& random : parallel_random_nodes) random.clear();
    parallel_safe = {{true, true}};
    scheduled_threads = 0;
    for(auto& player : infosets)
        for(auto& infoset : player) infoset.parallel_idx = -1;
}

void Environment::PrepareParallelSchedule() {
    const int threads = Parallel::GetThreads();
    if(scheduled_threads == threads) return;
    if(parallel_infosets.empty()) {
        // First-visit order is also the legacy Enumerate order. Own-player
        // parent depth, rather than history depth, groups independent infosets.
        for(auto* node : nodes) {
            if(node->player == 0 || node->is_terminal) continue;
            auto& infoset = infosets[node->player][node->infoset];
            if(infoset.parallel_idx >= 0) continue;
            infoset.parallel_idx = parallel_infosets.size();
            parallel_infosets.push_back({&infoset});
        }
        std::vector<std::vector<double>> widths;
        for(auto& entry : parallel_infosets) {
            auto& infoset = *entry.infoset;
            const auto& parents = infoset.parent_sequences[infoset.player];
            // Legacy file environments also accept imperfect-recall inputs.
            // Those do not necessarily form an own-player infoset forest.
            if(parents.size() > 1 || (!parents.empty() && parents[0] != infoset.parent))
                parallel_safe = {{false, false}};
            if(infoset.parent.first != 0) {
                auto& parent = infosets[infoset.player][infoset.parent.first];
                if(parent.parallel_idx < 0 || parent.parallel_idx >= infoset.parallel_idx) {
                    parallel_safe = {{false, false}};
                } else {
                    auto& parent_entry = parallel_infosets[parent.parallel_idx];
                    entry.depth = parent_entry.depth + 1;
                    parent_entry.children.push_back({&infoset, infoset.parent.second});
                }
            }
            if(parallel_layers.size() <= size_t(entry.depth)) parallel_layers.resize(entry.depth + 1);
            parallel_layers[entry.depth].infosets.push_back(infoset.parallel_idx);
            widths.push_back(EstimateWork(infoset, entry.work));
        }
        // Account for the actual child source widths copied and reduced.
        for(auto& entry : parallel_infosets) {
            auto& infoset = *entry.infoset;
            for(int phase=0; phase<2; ++phase) {
                const int status = GraphNode::NodeStatus::backward_node + phase;
                for(int i=infoset.graph.start_idx[status]; i<infoset.graph.start_idx[status+1]; ++i) {
                    if(!infoset.is_aggregator[i]) continue;
                    const auto& node = infoset.graph.graph_nodes[i];
                    if(node.operation->info[AggregateOperation::object] > 0.0)
                        for(const auto& child : entry.children)
                            entry.work[phase] += 2.0 * widths[child.first->parallel_idx][infoset.aggregator_dependency[i]];
                    else entry.work[phase] += 1.0;
                }
                entry.work[phase] = std::max(1.0, entry.work[phase]);
            }
        }
        if(!parallel_infosets.empty()) {
            const auto& local_graph = parallel_infosets[0].infoset->graph;
            for(int phase=0; phase<2; ++phase) {
                const int status = GraphNode::NodeStatus::backward_node + phase;
                std::unordered_set<int> written;
                for(int i=local_graph.start_idx[status]; i<local_graph.start_idx[status+1]; ++i)
                    written.insert(local_graph.graph_nodes[i].idx);
                for(int i=local_graph.start_idx[status]; i<local_graph.start_idx[status+1]; ++i) {
                    const auto& node = local_graph.graph_nodes[i];
                    if(!IsRandom(node)) continue;
                    // A dimension computed/mutated during this pass requires
                    // serial execution to preserve the legacy random stream.
                    if(node.dependency.size() != 1 || written.count(node.dependency[0]))
                        parallel_safe[phase] = false;
                    // Synthetic aggregator slots precede these nodes and their
                    // count differs with each infoset's action count.
                    parallel_random_nodes[phase].push_back(i - local_graph.start_idx[status]);
                }
            }
        }
    }
    for(auto& layer : parallel_layers) {
        for(int phase=0; phase<2; ++phase) {
            auto& boundaries = layer.boundaries[phase];
            const size_t count = layer.infosets.size();
            const size_t parts = std::min(size_t(threads), count);
            std::vector<double> prefix(count + 1, 0.0);
            for(size_t i=0; i<count; ++i)
                prefix[i+1] = prefix[i] + parallel_infosets[layer.infosets[i]].work[phase];
            boundaries.assign(1, 0);
            for(size_t part=1; part<parts; ++part) {
                const size_t low = boundaries.back() + 1, high = count - (parts - part);
                const double target = prefix.back() * double(part) / double(parts);
                auto next = std::lower_bound(prefix.begin()+low, prefix.begin()+high+1, target);
                size_t boundary = std::min(size_t(next-prefix.begin()), high);
                if(boundary > low && std::abs(prefix[boundary-1]-target) < std::abs(prefix[boundary]-target)) --boundary;
                boundaries.push_back(boundary);
            }
            boundaries.push_back(count);
        }
    }
    scheduled_threads = threads;
}

void Environment::GatherChildren(ParallelInfoset& entry, int status, bool sampled) {
    const bool backward = status == GraphNode::NodeStatus::backward_node;
    // Enumerate uses the precomputed order. Sampling may first visit a different
    // history of an infoset, so preserve this traversal's actual relative order.
    auto gather = [&](const auto& children) {
        for(size_t i=0; i<children.size(); ++i) {
            const auto& child = children[backward ? children.size()-1-i : i];
            if(child.first->update_order >= 0)
                entry.infoset->AggregateChildren(*child.first, child.second, status, is_color_to_update);
        }
    };
    if(sampled && entry.children.size() > 1) {
        auto children = entry.children;
        std::sort(children.begin(), children.end(), [](const auto& a, const auto& b) {
            return a.first->update_order < b.first->update_order;
        });
        gather(children);
    } else gather(entry.children);
}

bool Environment::UpdateParallel(const int& status, const int& current_traverse) {
    if(Parallel::GetThreads() == 1 || traverse_infoset.size() < 2) return false;
    const int phase = status - GraphNode::NodeStatus::backward_node;
    if(!parallel_safe[phase] || Is_Aggregate_Opponents) return false;
    const bool backward = phase == 0, sampled = current_traverse != Traverse::Enumerate;
    const auto& first_graph = traverse_infoset[0]->graph;
    if(first_graph.start_idx[status] == first_graph.start_idx[status+1]) {
        // Graph::Update advances this counter even for an empty pass.
        for(auto* infoset : traverse_infoset) ++infoset->graph.timestep;
        return true;
    }
    const auto& random_nodes = parallel_random_nodes[phase];
    // Validate every dimension before consuming any random numbers.
    for(auto* infoset : traverse_infoset)
        for(int offset : random_nodes) {
            const int i = infoset->graph.start_idx[status] + offset;
            const auto& node = infoset->graph.graph_nodes[i];
            const auto& size = infoset->results[node.dependency[0]];
            if(size.empty() || size[0].size != 1) return false;
        }
    auto clear_random = [&] {
        if(!random_nodes.empty())
            for(auto* infoset : traverse_infoset) infoset->graph.prepared_random.clear();
    };
    try {
        for(size_t t=0; !random_nodes.empty() && t<traverse_infoset.size(); ++t) {
            auto& infoset = *traverse_infoset[backward ? traverse_infoset.size()-1-t : t];
            for(int offset : random_nodes) {
                const int i = infoset.graph.start_idx[status] + offset;
                auto& node = infoset.graph.graph_nodes[i];
                if(is_color_to_update[node.color])
                    node.operation->Execute(infoset.graph.prepared_random[i], {&infoset.results[node.dependency[0]][0]});
            }
        }
        // Forward child aggregates read values from before the entire forward
        // pass. Each destination owns its buffers, including unvisited parents.
        if(!backward)
            for(auto& layer : parallel_layers) {
                const auto& boundaries = layer.boundaries[phase];
                Parallel::Run(boundaries.size()-1, [&](size_t part) {
                    for(size_t i=boundaries[part]; i<boundaries[part+1]; ++i)
                        GatherChildren(parallel_infosets[layer.infosets[i]], status, sampled);
                });
            }
        for(size_t depth=0; depth<parallel_layers.size(); ++depth) {
            auto& layer = parallel_layers[backward ? parallel_layers.size()-1-depth : depth];
            const auto& boundaries = layer.boundaries[phase];
            Parallel::Run(boundaries.size()-1, [&](size_t part) {
                for(size_t i=boundaries[part]; i<boundaries[part+1]; ++i) {
                    auto& entry = parallel_infosets[layer.infosets[i]];
                    if(backward) GatherChildren(entry, status, sampled);
                    if(entry.infoset->update_order < 0) continue;
                    AggregateInformation(*entry.infoset, true, status);
                    entry.infoset->UpdateGraph(status, is_color_to_update);
                }
            }); // Layer barrier: all children/parents are complete before advancing.
        }
    } catch(...) {
        clear_random();
        throw;
    }
    clear_random();
    return true;
}
