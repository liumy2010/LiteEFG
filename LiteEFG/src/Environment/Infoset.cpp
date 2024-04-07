#include "Environment/Infoset.h"

#include "Basic/BasicFunction.h"
#include "Computation/Static.h"

#include <algorithm>
#include <string>
#include <stdexcept>

Node::Node(const int& player_, const int& infoset_, const int& player_num_) : player(player_), infoset(infoset_), player_num{player_num_} {
    for(int i=0; i<=player_num; i++){
        parent_infoset.push_back(std::make_pair(0, 0));
    }
}

double Node::GetUtility(const int& player){
    return 0.0;
}

void Node::Preprocessing(std::vector<Node*>& nodes, const int& player_num_){
    // Preprocessing
    nodes.insert(nodes.begin(), new Node(0, 0, player_num_)); 
    // virtual root node with only one child to the original root
    // the reach probability would easier to calculate
    // Moreover, the game always starts with player 0, the chance node

    nodes[0] -> chance = Vector(1, 1.0);
    nodes[0] -> next_node = std::vector<int>(1, 1);
    nodes[0] -> parent = std::make_pair(0, 0);
    nodes[0] -> reach = Vector(player_num_+1, 1.0);
    for(int player=0; player<=player_num_; player++)
        nodes[0] -> parent_infoset[player] = std::make_pair(0, 0);
    for(int i=1; i<nodes.size(); i++)
        for(int j=0; j<nodes[i] -> next_node.size(); j++)
            nodes[i] -> next_node[j]++;

    int idx = 0;
    std::vector<std::vector<std::pair<int, int>>> infoset_order;
    infoset_order.clear();
    for(int i=0;i<=player_num_;i++) infoset_order.push_back(std::vector<std::pair<int, int>>()), infoset_order[i].clear();
    for (auto* node : nodes){
        node -> idx = idx++;
        node -> is_terminal = (node -> next_node.size() == 0);
        

        if(node->player && !node -> is_terminal) {
            if(node -> player != 0 && node -> infoset == 0)
                throw std::invalid_argument("Infoset index should start from 1, node idx: " + std::to_string(node->idx-1) + " does not satisfy it");
            infoset_order[node->player].push_back(std::make_pair(node->infoset, node->idx));
        } else { // chance node and terminal ndoes
            node -> infoset = 0;
        }
    }
    for(int i=1;i<=player_num_;i++){
        std::sort(infoset_order[i].begin(), infoset_order[i].end());
        for(int j=0, num_infoset=1; j<infoset_order[i].size(); j++){
            nodes[infoset_order[i][j].second] -> infoset = num_infoset;
            if(j<infoset_order[i].size()-1 && infoset_order[i][j].first != infoset_order[i][j+1].first)
                num_infoset++;
        }
    }

    for (auto* node : nodes){
        for(int i=0; i<node -> next_node.size(); i++){
            nodes[node -> next_node[i]]->parent = std::make_pair(node->idx, i);
        }

        for (int player=0; player<=player_num_; player++){
            if(nodes[node -> parent.first]->player == player)
                node -> parent_infoset[player] = std::make_pair(nodes[node -> parent.first] -> infoset, node -> parent.second);
            else
                node -> parent_infoset[player] = nodes[node -> parent.first] -> parent_infoset[player];
        }
    }
}

Infoset::Infoset() {
    /*if traverse == "Enumerate"{
        prefix_graph.AddOperation(Constants::default_counterfactual_name, std::make_shared<Aggregate>AggregateOperation(), {"counterfactual_ev"});
        prefix_graph.AddOperation(Constants::default_counterfactual_name, std::make_shared<DotOperation>(), {Constants::default_counterfactual_name, strategy});
    } else if traverse == "Outcome"{
    } else if traverse == "External"{
    } else{
        throw std::invalid_argument("Invalid Traverse");
    }
    prefix_graph.AddOperation(Constants::default_counterfactual_ev_name, )*/

    children.clear();
    parent = std::make_pair(0, 0); // root infoset
}

void Infoset::UpdateGraph(const int& status) {
    graph.Update(results, status);
}

void Infoset::InitializeAggregator(){
    for(int i=0; i<graph.graph_nodes.size(); i++) {
        if(graph.graph_nodes[i].operation!=NULL && graph.graph_nodes[i].operation->name == "Aggregate_children"){
            graph.graph_nodes[i].dependency.resize(children.size() + 1);
            graph.graph_nodes[i].dependency.back() = graph.graph_nodes[i].dependency[0];
            for(int action=0; action<children.size(); action++){
                graph.graph_nodes[i].dependency[action] = graph.graph_nodes.size();
                graph.graph_nodes.push_back(GraphNode(graph.graph_nodes.size(), {}, NULL, GraphNode::NodeStatus::smallest_status));
            }
        } else if(graph.graph_nodes[i].operation!=NULL && graph.graph_nodes[i].operation->name == "Aggregate_parent"){
            graph.graph_nodes[i].dependency.resize(2);
            graph.graph_nodes[i].dependency.back() = graph.graph_nodes[i].dependency[0];
            graph.graph_nodes[i].dependency[0] = graph.graph_nodes.size();
            graph.graph_nodes.push_back(GraphNode(graph.graph_nodes.size(), {}, NULL, GraphNode::NodeStatus::smallest_status));
        }
    }

    results.resize(graph.graph_nodes.size(), {});
    aggregator_dependency.resize(graph.graph_nodes.size(), -1);
    aggregator_type.resize(graph.graph_nodes.size(), -1);
    graph.Initialize();
    for(int i=0; i<graph.graph_nodes.size(); i++) {
        GraphNode& graph_node = graph.graph_nodes[i];
        if(graph_node.operation!=NULL && Basic::IsPrefixString(graph_node.operation->name, "Aggregate")){
            aggregator_dependency[i] = graph_node.dependency.back();
            aggregator_type[i] = (graph_node.operation->name == "Aggregate_children") ? 0 : 1;
            graph_node.dependency.pop_back();
        }
    }
}

void Infoset::AggregateChildren(Infoset& child_infoset, const int& status) {
    for(int i=graph.start_idx[status]; i<graph.start_idx[status+1]; i++) if(graph.graph_nodes[i].status == status && aggregator_type[i]==0){
        GraphNode& graph_node = graph.graph_nodes[i];
        int dependency = aggregator_dependency[i];
        results[graph_node.dependency[child_infoset.parent.second]][0].Concat(child_infoset.results[dependency][0]);
    }
}

void Infoset::AggregateParent(Infoset& parent_infoset, const int& status) {
    if(parent.first==0) return;
    for(int i=graph.start_idx[status]; i<graph.start_idx[status+1]; i++) if(graph.graph_nodes[i].status == status && aggregator_type[i]==1){
        GraphNode& graph_node = graph.graph_nodes[i];
        int dependency = aggregator_dependency[i];
        results[graph_node.dependency[0]][0].Concat(parent_infoset.results[dependency][0]);
    }
}


void Infoset::InitializeGraph(const double& reach){
    results[GraphNode::NodeIdx::utility][0].Set(0.0);
    results[GraphNode::NodeIdx::opponent_reach_prob][0][0] = 0.0;
    results[GraphNode::NodeIdx::reach_prob][0][0] = reach;
    for(int i=0; i<graph.graph_nodes.size(); i++) {
        GraphNode& graph_node = graph.graph_nodes[i];
        if(aggregator_dependency[i]!=-1){
            for(int action=0; action<graph.graph_nodes[i].dependency.size(); action++){
                int idx = graph_node.dependency[action];
                if(results[idx].size() == 0){
                    results[idx].push_back(Vector());
                }
                results[idx][0].Resize(0);
                /*if(aggregator_type[i]==0 && !action_activated[action]){ // cannot use action_activated. sometimes although activated, no children
                    results[idx][0].Resize(1); // No infoset child under current action
                    results[idx][0][0] = graph_node.operation->info[0];
                } else if(aggregator_type[i]==1 && parent.first==0){
                    results[idx][0].Resize(1); // No infoset parent
                    results[idx][0][0] = graph_node.operation->info[0];
                } else{
                    results[idx][0].Resize(0);
                }*/
            }
        }
    }
}

Vector Infoset::GetResult(const int& opIndex) {
    if (opIndex >= results.size()) {
        throw std::runtime_error("GetResult cannot acceed the results length");
    }
    return results[opIndex][0];
}