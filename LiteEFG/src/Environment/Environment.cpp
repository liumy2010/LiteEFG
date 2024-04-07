#include "Environment/Environment.h"

#include "Computation/Operations.h"
#include "Basic/BasicFunction.h"

#include <string>
#include <stdexcept>

Environment::Environment(const int& player_num_, const std::string& traverse_)
    : player_num{player_num_} {
    
    if(traverse_ == "Enumerate") traverse = Traverse::Enumerate;
    else if(traverse_ == "Outcome") traverse = Traverse::Outcome;
    else if(traverse_ == "External") traverse = Traverse::External;
    else throw std::invalid_argument("Only support [Enumerate, Outcome, External] for traverse");
}

void Environment::GetGraph(const Graph& graph_){
    graph = graph_;
    if(!Flags_Initialized){
        Initialize();
    }

    for(int player=1; player<=player_num;player++){
        for(int i=infosets[player].size()-1; i>=0; i--){
            Infoset& infoset = infosets[player][i];
            infoset.graph = graph;
            infoset.InitializeAggregator();
        }
    }
    for(int player=1; player<=player_num;player++){
        for(int i=0; i<infosets[player].size(); i++){
            Infoset& infoset = infosets[player][i];
            infoset.InitializeGraph(0.0);
        }
    }
    for(int player=1; player<=player_num;player++){
        for(int i=infosets[player].size()-1; i>0; i--){
            Infoset& infoset = infosets[player][i];
            infoset.AggregateParent(infosets[player][infoset.parent.first], GraphNode::NodeStatus::static_backward_node);
            infoset.UpdateGraph(GraphNode::NodeStatus::static_backward_node);
            infosets[player][infoset.parent.first].AggregateChildren(infoset, GraphNode::NodeStatus::static_backward_node);
        }
        for(int i=infosets[player].size()-1; i>0; i--){
            Infoset& infoset = infosets[player][i];
            infosets[player][infoset.parent.first].AggregateChildren(infoset, GraphNode::NodeStatus::static_forward_node);
        }
        for(int i=1; i<infosets[player].size(); i++){
            Infoset& infoset = infosets[player][i];
            infoset.AggregateParent(infosets[player][infoset.parent.first], GraphNode::NodeStatus::static_forward_node);
            infoset.UpdateGraph(GraphNode::NodeStatus::static_forward_node);
        }
    }
}

double Environment::GetProb(Node* node, const int& strategy_node_idx, const int& action){
    if(node -> player == 0)
        return (node -> chance)[action];
    Infoset* infoset = &infosets[node -> player][node -> infoset];
    if(action >= infoset -> results[strategy_node_idx][0].size){
        throw std::invalid_argument("action out of range, please check the strategy feed into env.Update()");
    }
    return infoset->results[strategy_node_idx][0][action];
}

Vector* Environment::GetProb(Node* node, const int& strategy_node_idx){
    if(node -> player == 0) return &(node -> chance);
    Infoset* infoset = &infosets[node -> player][node -> infoset];
    return &infoset->results[strategy_node_idx][0];
}

void Environment::Initialize(){

    Node::Preprocessing(nodes, player_num);

    for(int i=0;i<=player_num;i++){
        infosets.push_back(std::vector<Infoset>());
        infosets[i].push_back(Infoset());
        infosets[i][0].reach = 1.0; // root infoset
        infosets[i][0].children.push_back(std::vector<int>());
    }

    for (auto* node : nodes){
        while(infosets[node -> player].size() <= node -> infoset) infosets[node->player].push_back(Infoset());

        Infoset& infoset = infosets[node->player][node -> infoset];
        while(infoset.children.size() < node -> next_node.size())
            infoset.children.push_back(std::vector<int>());
        if(infoset.children.size() == 0 && !node -> is_terminal)
            throw std::invalid_argument("infoset with no children should only contain terminal nodes");
        // action set is the same for all nodes in the same infoset, otherwise they can be differed from others
        // we let the action set to be at least 1. If it is empty (such as terminal nodes), we force it to 1

        infoset.parent = node -> parent_infoset[node -> player];
        infoset.player = node -> player;
    }

    for(int player=1;player<=player_num;player++){
        for(int i=1;i<infosets[player].size();++i){
            infosets[player][infosets[player][i].parent.first].children[infosets[player][i].parent.second].push_back(i);
        }
    }

    for (int player=1; player<=player_num; player++){
        for(int i=0; i<infosets[player].size(); i++){
            Infoset& infoset = infosets[player][i];
            infoset.results.resize(GraphNode::num_nodes, {});
            infoset.size = 0;
            infoset.results[GraphNode::NodeIdx::action_set_size] = {Vector(1, double(infoset.children.size()))}; 
            infoset.results[GraphNode::NodeIdx::utility] = {Vector(infoset.children.size(), 0.0)};
            infoset.results[GraphNode::NodeIdx::subtree_size] = {Vector(infoset.children.size(), 0.0)};
            infoset.results[GraphNode::NodeIdx::reach_prob] = {Vector(1, 0.0)};
            infoset.results[GraphNode::NodeIdx::opponent_reach_prob] = {Vector(1, 0.0)};
        }
    }
    for(int player=1;player<=player_num;player++){
        for(int i= infosets[player].size()-1; i>0; i--){
            Infoset& infoset = infosets[player][i];
            for(int action=0; action<infoset.children.size(); ++action){
                if(infoset.children[action].size() == 0) infoset.size += 1;
                infoset.results[GraphNode::NodeIdx::subtree_size][0][action] ++;
            }
            infosets[player][infoset.parent.first].size += infoset.size;
            infosets[player][infoset.parent.first].results[GraphNode::NodeIdx::subtree_size][0][infoset.parent.second] += infoset.size;
        }
    }

    // Initialize sequence form strategy_names
    while(sequence_form_strategies.size() <= player_num){
        int player = sequence_form_strategies.size();
        sequence_form_strategies.push_back(SequenceForm(&infosets[player]));
    }
    Flags_Initialized = true;
}

bool CheckValidPlayer(const int& player, const int& upd_player){
    return player != 0 && (player == upd_player || upd_player == -1);
}

bool CheckValidNode(Node* node, const int& upd_player){
    return (CheckValidPlayer(node -> player, upd_player) && !node -> is_terminal);
}

void Environment::UpdateTraverse(const int& upd_player){

    for(int t=traverse_order.size()-1; t>=0; t--) if(CheckValidNode(traverse_order[t], upd_player)){
        int player = traverse_order[t] -> player;
        infosets[player][traverse_order[t] -> infoset].first_visited = t;
    }

    traverse_infoset.clear();
    for(int t=0; t<traverse_order.size(); t++) if(CheckValidNode(traverse_order[t], upd_player)){
        Node* node = traverse_order[t];
        Infoset& infoset = infosets[node -> player][node -> infoset];
        if(t == infoset.first_visited){
            // Only update the graph once when the player is not chance.
            infoset.InitializeGraph(node -> reach[node -> player]);
            traverse_infoset.push_back(&infoset);
        }
    }
    for(int player=1; player<=player_num; player++) if(CheckValidPlayer(player, upd_player)){
        infosets[player][0].InitializeGraph(1.0);
    }

    double reach_prob_cum_mul[player_num+2];
    for(int t=traverse_order.size()-1; t>=0; t--) { 
        // Not just the upd_player's node should be visited. Because some terminal nodes belong to non-upd_players
        Node* node = traverse_order[t];
        
        reach_prob_cum_mul[player_num+1] = 1.0;
        for(int p=player_num; p>=0; --p) reach_prob_cum_mul[p] = reach_prob_cum_mul[p+1] * node -> reach[p];
        double cum_mul = node -> reach[0];
        for(int p=1; p<=player_num; ++p) {
            if(CheckValidPlayer(p, upd_player)){
                auto parent = node -> parent_infoset[p];
                infosets[p][parent.first].results[GraphNode::NodeIdx::utility][0][parent.second] += node -> GetUtility(p) * 
                                                                                                        ((traverse==Traverse::Enumerate) ? cum_mul * reach_prob_cum_mul[p+1] : 1.0);
                if(p == node -> player)
                    infosets[node->player][node->infoset].results[GraphNode::NodeIdx::opponent_reach_prob][0][0] += cum_mul * reach_prob_cum_mul[p+1];
            }
            
            cum_mul *= node -> reach[p];
        }
    }

    for(int t=traverse_infoset.size()-1; t>=0; t--){
        Infoset& infoset = *traverse_infoset[t];
        infoset.AggregateParent(infosets[infoset.player][infoset.parent.first], GraphNode::NodeStatus::backward_node);
        infoset.UpdateGraph(GraphNode::NodeStatus::backward_node);
        infosets[infoset.player][infoset.parent.first].AggregateChildren(infoset, GraphNode::NodeStatus::backward_node);
    }

    for(int t=0; t<traverse_infoset.size(); t++){
        Infoset& infoset = *traverse_infoset[t];
        infosets[infoset.player][infoset.parent.first].AggregateChildren(infoset, GraphNode::NodeStatus::forward_node);
    }

    for(int t=0; t<traverse_infoset.size(); t++){
        Infoset& infoset = *traverse_infoset[t];
        infoset.AggregateParent(infosets[infoset.player][infoset.parent.first], GraphNode::NodeStatus::forward_node);
        infoset.UpdateGraph(GraphNode::NodeStatus::forward_node);
    }
}

void Environment::Update(const GraphNode& strategy_node, const int& upd_player){
    std::vector<GraphNode> strategy_nodes;
    for(int i=1;i<=player_num;i++) strategy_nodes.push_back(strategy_node);
    Update(strategy_nodes, upd_player);
}

void Environment::Update(std::vector<GraphNode> strategy_nodes, const int& upd_player){
    /*
        strategy_name is the name of the variable in results that contains the strategy_name to traverse the tree
        traverse is the method to traverse the tree
        upd_player is the player to update the graph. If -1, update all players
    */
    if(!Flags_Initialized){
        Initialize();
    }

    strategy_nodes.insert(strategy_nodes.begin(), strategy_nodes[0]); // chance player, just a placeholder

    traverse_order.clear();
    nodes[0] -> reach.Resize(player_num+1); // players include chance player, with chance player idx = 0
    nodes[0] -> reach.Set(1.0);
    if(traverse == Traverse::Enumerate){
        traverse_order.resize(nodes.size());
        for (int i=0; i<nodes.size(); ++i){
            Node* node = nodes[i];
            node->reach = nodes[node -> parent.first] -> reach;
            int player = nodes[node->parent.first]->player;
            node->reach[player] *= GetProb(nodes[node->parent.first], strategy_nodes[player].idx, node->parent.second);
            traverse_order[i] = node;
        }
        UpdateTraverse(upd_player);
    } else if (traverse == Traverse::Outcome){
        traverse_order.clear();
        traverse_order.push_back(nodes[0]);
        for (int i=0; i<traverse_order.size(); i++){
            if(traverse_order[i] -> next_node.size() == 0) continue;
            int player = traverse_order[i]->player;
            int action = Basic::Sample(GetProb(traverse_order[i], strategy_nodes[player].idx));
            Node* next_node = nodes[traverse_order[i]->next_node[action]];
            next_node->reach = traverse_order[i]->reach;
            next_node->reach[player] *= GetProb(traverse_order[i], strategy_nodes[player].idx, action);

            traverse_order.push_back(next_node);
        }
        UpdateTraverse(upd_player);
    } else if (traverse == Traverse::External){
        for(int player=1; player<=player_num; player++) if(CheckValidPlayer(player, upd_player)){
            traverse_order.clear();
            traverse_order.push_back(nodes[0]); // same as above
            for (int i=0; i<traverse_order.size(); i++){
                if(traverse_order[i] -> next_node.size() == 0) continue;
                if(traverse_order[i]->player == player){
                    for(int action=0; action<traverse_order[i]->next_node.size(); ++action){
                        Node* next_node = nodes[traverse_order[i]->next_node[action]];
                        next_node -> reach = traverse_order[i]->reach;
                        
                        next_node -> reach[player] *= GetProb(traverse_order[i], strategy_nodes[player].idx, action);
                        traverse_order.push_back(next_node);
                    }
                } else{
                    int player = traverse_order[i]->player;
                    int action = Basic::Sample(GetProb(traverse_order[i], strategy_nodes[player].idx));
                    Node* next_node = nodes[traverse_order[i]->next_node[action]];
                    next_node->reach = traverse_order[i]->reach;
                    next_node->reach[player] *= GetProb(traverse_order[i], strategy_nodes[player].idx, action);
                    
                    traverse_order.push_back(next_node);
                }
            }
            UpdateTraverse(player);
        }
    } else{
        throw std::invalid_argument("Invalid Traverse");
    }
}

void Environment::UpdateStrategy(const GraphNode& strategy_node, const bool& update_best){
    std::vector<GraphNode> strategy_nodes;
    for(int i=1;i<=player_num;i++) strategy_nodes.push_back(strategy_node);
    UpdateStrategy(strategy_nodes, update_best);
}

void Environment::UpdateStrategy(const std::vector<GraphNode>& strategy_nodes, const bool& update_best){
    if(strategy_nodes.size() != player_num){
        throw std::invalid_argument("strategy_names.size() needs to match player_num");
    }
    double exploitability = Constants::INF;
    if(update_best){
        auto exploitability_list = Exploitability(strategy_nodes);
        exploitability = 0.0;
        for(int i=0; i<exploitability_list.size(); i++) exploitability += exploitability_list[i];
    } else{
        if(!Flags_Initialized){
            Initialize();
        }

        for(int player=1; player<=player_num; player++){
            sequence_form_strategies[player].GetSequenceFormStrategy(strategy_nodes[player-1].idx);
            sequence_form_strategies[player].IsSequenceForm(sequence_form_strategies[player].strategy); // Check validility
        }
    }
    for(int i=1;i<=player_num;i++){
        sequence_form_strategies[i].UpdateStrategy(strategy_nodes[i-1].idx, exploitability);
    }
}

void Environment::GetGradient(const std::vector<GraphNode>& strategy_nodes, const std::string& type_name){
    if(strategy_nodes.size() != player_num){
        throw std::invalid_argument("strategy_names.size() needs to match player_num");
    }
    if(!Flags_Initialized){
        Initialize();
    }

    for(int player=1; player<=player_num; player++){
        sequence_form_strategies[player].GetSequenceFormStrategy(strategy_nodes[player-1].idx, type_name);
        sequence_form_strategies[player].IsSequenceForm(sequence_form_strategies[player].strategy); // Check validility
    }

    double reach_prob_cum_mul[player_num+2];
    for (auto* node : nodes){
        node -> reach = nodes[node -> parent.first] -> reach;

        int parent_player = nodes[node->parent.first]->player;
        if(parent_player == 0)
            node->reach[parent_player] *= GetProb(nodes[node->parent.first], 0, node->parent.second);
        else{
            int idx = sequence_form_strategies[parent_player].GetIdx(node->parent_infoset[parent_player].first, node->parent_infoset[parent_player].second);
            node->reach[parent_player] = sequence_form_strategies[parent_player].strategy[idx];
        }
        
        if(node -> is_terminal){
            reach_prob_cum_mul[player_num+1] = 1.0;
            for(int p=player_num; p>=0; --p) reach_prob_cum_mul[p] = reach_prob_cum_mul[p+1] * node -> reach[p];
            double cum_mul = node -> reach[0];
            for(int p=1; p<=player_num; ++p){
                int idx = sequence_form_strategies[p].GetIdx(node->parent_infoset[p].first, node->parent_infoset[p].second);
                sequence_form_strategies[p].gradient[idx] += node -> GetUtility(p) * cum_mul * reach_prob_cum_mul[p+1];
                cum_mul *= node -> reach[p];
            }
        }
    }
}

std::vector<double> Environment::Utility(const GraphNode& strategy_node, const std::string& type_name){
    std::vector<GraphNode> strategy_nodes;
    for(int i=1;i<=player_num;i++) strategy_nodes.push_back(strategy_node);
    return Utility(strategy_nodes, type_name);
}

std::vector<double> Environment::Utility(const std::vector<GraphNode>& strategy_nodes, const std::string& type_name){
    GetGradient(strategy_nodes, type_name);
    std::vector<double> utility;
    for(int i=1;i<=player_num;i++) utility.push_back(sequence_form_strategies[i].GetUtility());
    return utility;
}

std::vector<double> Environment::Exploitability(const GraphNode& strategy_node, const std::string& type_name){
    std::vector<GraphNode> strategy_nodes;
    for(int i=1;i<=player_num;i++) strategy_nodes.push_back(strategy_node);
    return Exploitability(strategy_nodes, type_name);
}

std::vector<double> Environment::Exploitability(const std::vector<GraphNode>& strategy_nodes, const std::string& type_name){
    GetGradient(strategy_nodes, type_name);
    std::vector<double> exploitability;
    for(int i=1;i<=player_num;i++) exploitability.push_back(sequence_form_strategies[i].GetExploitability());
    return exploitability;
}

std::vector<double> Environment::GetSequenceFormStrategy(const int& player, const GraphNode& strategy_node){
    sequence_form_strategies[player].GetSequenceFormStrategy(strategy_node.idx);
    std::vector<double> ret_strategy = std::vector<double>(sequence_form_strategies[player].strategy.size, 0.0);
    for(int i=0; i<ret_strategy.size(); ++i)
        ret_strategy[i] = sequence_form_strategies[player].strategy[i];
    return ret_strategy;
}