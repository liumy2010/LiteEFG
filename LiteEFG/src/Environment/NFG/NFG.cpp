#include "Environment/NFG/NFG.h"

NFGNode::NFGNode(const int& player_, const int& infoset_, const int& player_num_) : Node(player_, infoset_, player_num_) {
    actions.clear();
    utility = std::vector<double>(player_num+1, 0.0);
}

double NFGNode::GetUtility(const int& player){
    return utility[player];
}

NFG::NFG(const int& player_num_, const std::vector<Tensor>& utility_matrices_, const std::string& traverse_)
        : Environment(player_num_, traverse_){

    utility_matrices = utility_matrices_;

    nfg_nodes.push_back(NFGNode(1, 1, player_num));

    for(int i=0; i<nfg_nodes.size(); ++i){
        if(nfg_nodes[i].actions.size() == player_num) continue; // terminal node
        int player = nfg_nodes[i].player;
        for(int a=0; a<utility_matrices[0].shape[player-1]; ++a){
            nfg_nodes.push_back((player==player_num) ? NFGNode(1, 2, player_num) : NFGNode(player+1, 1, player_num));
            int idx = nfg_nodes.size()-1;
            nfg_nodes[idx].actions = nfg_nodes[i].actions;
            nfg_nodes[idx].actions.push_back(a);
            
            nfg_nodes[i].next_node.push_back(idx);

            if(player == player_num){
                for(int p=1;p<=player_num;++p){
                    nfg_nodes[idx].utility[p] = utility_matrices[p-1][nfg_nodes[idx].actions];
                }
                /*for(int p=2; p<=player_num; ++p){
                    nfg_nodes.push_back(NFGNode(p, nfg_nodes[idx].actions[p-1]+2, player_num));
                    nfg_nodes[nfg_nodes.size()-1].actions = nfg_nodes[idx].actions;
                    nfg_nodes[idx].next_node.push_back(nfg_nodes.size()-1);
                    idx = nfg_nodes.size()-1;
                    nfg_nodes[idx].utility = utility_matrices[p-1][nfg_nodes[idx].actions];
                }*/
            }
        }
    }

    for(int i=0; i<nfg_nodes.size(); ++i){
        nodes.push_back(&nfg_nodes[i]);
    }
}

void NFG::Initialize(){
    Environment::Initialize();
}