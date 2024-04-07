#include "Graph.h"

#include "Data/Vector.h"

#include <algorithm>
#include <stdexcept>

bool GraphNode_cmp(const GraphNode& a, const GraphNode& b) {
    if(a.status != b.status) return a.status < b.status; // order: static, forward, backward
    return a.order < b.order;
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

void Graph::ForwardNode(const bool& is_static) {
    GraphNode::graph_status = (is_static) ? GraphNode::NodeStatus::static_forward_node : GraphNode::NodeStatus::forward_node;
}

void Graph::BackwardNode(const bool& is_static) {
    GraphNode::graph_status = (is_static) ? GraphNode::NodeStatus::static_backward_node : GraphNode::NodeStatus::backward_node;
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

void Graph::Update(std::vector<std::vector<Vector>>& results, const int& status) {
    if(status >= GraphNode::NodeStatus::status_num) throw std::runtime_error("Invalid status");
    for(int i = start_idx[status]; i < start_idx[status+1]; ++i) if(graph_nodes[i].status == status){
        Execute(results, i);
    }
    timestep++;
}