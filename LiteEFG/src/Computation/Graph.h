#ifndef GRAPH_H
#define GRAPH_H

#include "Data/Vector.h"
#include "Operations.h"
#include "GraphNode.h"

#include <pybind11/pybind11.h>
#include <unordered_map>
#include <functional>
#include <map>

class GraphNodeStatus {
public:
    int graph_status, color;
    GraphNodeStatus();
    GraphNodeStatus* Enter();
    void Exit(pybind11::args);
};

class ForwardNodeStatus : public GraphNodeStatus {
public:
    ForwardNodeStatus(const bool& is_static=false, const int& color_=0);
};

class BackwardNodeStatus : public GraphNodeStatus {
public:
    BackwardNodeStatus(const bool& is_static=false, const int& color_=0);
};

class Graph {
public:
    std::vector<std::string> order;
    std::vector<Vector*> inputs;

    GraphNode utility, opponent_reach_prob, reach_prob, action_set_size, subtree_size;
    std::vector<GraphNode> graph_nodes;
    //std::vector<int> position;

    int timestep=0, start_idx[GraphNode::NodeStatus::status_num+1];

    Graph();

    void Initialize();
    int UpdateColorMapping(std::map<int, int>& color_mapping);
    void Execute(std::vector<std::vector<Vector>>& results, const int& opName);
    void Update(std::vector<std::vector<Vector>>& results, const int& status, const std::vector<bool>& is_color_to_update);
};

#endif
