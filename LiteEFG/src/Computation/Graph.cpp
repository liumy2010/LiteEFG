#include "Graph.h"

#include "Data/Vector.h"

#include <algorithm>
#include <stdexcept>

bool GraphNode_cmp(const GraphNode& a, const GraphNode& b) {
    if(a.status != b.status) return a.status < b.status; // order: static, forward, backward
    return a.order < b.order;
}

GraphNodeStatus::GraphNodeStatus(){}

GraphNodeStatus* GraphNodeStatus::Enter() {
    GraphNode::graph_status = graph_status;
    GraphNode::graph_color = color;
    return this;
}

void GraphNodeStatus::Exit(pybind11::args) {
    GraphNode::graph_status = GraphNode::NodeStatus::smallest_status;
    GraphNode::graph_color = 0;
}

ForwardNodeStatus::ForwardNodeStatus(const bool& is_static, const int& color_) : GraphNodeStatus() {
    graph_status = (is_static) ? GraphNode::NodeStatus::static_forward_node : GraphNode::NodeStatus::forward_node;
    color = color_;
}

BackwardNodeStatus::BackwardNodeStatus(const bool& is_static, const int& color_) : GraphNodeStatus() {
    graph_status = (is_static) ? GraphNode::NodeStatus::static_backward_node : GraphNode::NodeStatus::backward_node;
    color = color_;
}

Graph::Graph() {
    GraphNode::num_nodes = GraphNode::NodeIdx::start;
    GraphNode::graph_status = GraphNode::NodeStatus::smallest_status;
    graph_nodes.clear();

    for(int i=0; i<GraphNode::num_nodes; ++i)
        graph_nodes.push_back(GraphNode(i, {}, NULL, GraphNode::NodeStatus::smallest_status)); // placeholder for some predefined variables
    utility = graph_nodes[GraphNode::NodeIdx::utility];
    opponent_reach_prob = graph_nodes[GraphNode::NodeIdx::opponent_reach_prob];
    reach_prob = graph_nodes[GraphNode::NodeIdx::reach_prob];
    action_set_size = graph_nodes[GraphNode::NodeIdx::action_set_size];
    subtree_size = graph_nodes[GraphNode::NodeIdx::subtree_size];

    GraphNode::graph_nodes = &graph_nodes;
}

void Graph::Initialize() {
    std::sort(graph_nodes.begin(), graph_nodes.end(), GraphNode_cmp);
    for(int i=0; i<GraphNode::NodeStatus::status_num; ++i) start_idx[i] = -1;
    start_idx[GraphNode::NodeStatus::status_num] = graph_nodes.size();
    for(int i=graph_nodes.size()-1; i>=GraphNode::NodeIdx::start; --i){
        start_idx[graph_nodes[i].status] = i;
    }
    for(int i=GraphNode::NodeStatus::status_num-1; i>=0; --i) {
        if(start_idx[i] == -1) start_idx[i] = start_idx[i+1];
    }
}

int Graph::UpdateColorMapping(std::map<int, int>& color_mapping) {
    color_mapping.clear();
    
    std::vector<int> color_list;
    for(int i=0; i<graph_nodes.size(); ++i) {
        color_list.push_back(graph_nodes[i].color);
    }
    std::sort(color_list.begin(), color_list.end());

    int num_colors = 0;
    for(int i=0; i<color_list.size(); ++i) {
        if(i==0 || color_list[i] != color_list[i-1]){
            color_mapping[color_list[i]] = num_colors++;
        }
    }
    for(int i=0; i<graph_nodes.size(); ++i) {
        graph_nodes[i].color = color_mapping[graph_nodes[i].color];
    }
    return num_colors;
}

void Graph::Execute(std::vector<std::vector<Vector>>& results, const int& opIndex) {
    if(graph_nodes[opIndex].operation == NULL) return;
    inputs.resize(graph_nodes[opIndex].dependency.size());

    for(int i=0; i<inputs.size(); ++i) {
        inputs[i] = &results[graph_nodes[opIndex].dependency[i]][0]; // Be careful whether graph_nodes[opIndex].idx == opIndex
    }

    int idx = graph_nodes[opIndex].idx;
    if(results[idx].size() == 0) {
        results[idx].push_back(Vector());
    }
    graph_nodes[opIndex].operation->Execute(results[idx][0], inputs); // Store result for future use
}

void Graph::Update(std::vector<std::vector<Vector>>& results, const int& status, const std::vector<bool>& is_color_to_update) {
    if(status >= GraphNode::NodeStatus::status_num) throw std::runtime_error("Invalid status");
    for(int i = start_idx[status]; i < start_idx[status+1]; ++i) {
        if(graph_nodes[i].status == status && is_color_to_update[graph_nodes[i].color]) {
            Execute(results, i);
        }
    }
    timestep++;
}