#ifndef GRAPH_H
#define GRAPH_H

#include "Data/Vector.h"
#include "Operations.h"
#include "GraphNode.h"

#include <unordered_map>
#include <functional>

class Graph {
public:
    std::vector<std::string> order;
    std::vector<Vector*> inputs;

    GraphNode utility, opponent_reach_prob, reach_prob, action_set_size, subtree_size;
    std::vector<GraphNode> graph_nodes;
    //std::vector<int> position;

    int timestep=0, start_idx[GraphNode::NodeStatus::status_num+1];

    Graph();

    void ForwardNode(const bool& is_static=false);
    void BackwardNode(const bool& is_static=false);

    void Initialize();
    void Execute(std::vector<std::vector<Vector>>& results, const int& opName);
    void Update(std::vector<std::vector<Vector>>& results, const int& status=GraphNode::NodeStatus::backward_node);
};

#endif
