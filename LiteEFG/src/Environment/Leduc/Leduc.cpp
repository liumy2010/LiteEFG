#include "Environment/Leduc/Leduc.h"

#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <cstring>

bool Card::operator==(const Card& card){
    return private_card == card.private_card;
}

bool Card::operator<(const Card& card){
    /*
        Assume card != *this
    */
   if((private_card==public_card) != (card.private_card==card.public_card))
       return (private_card==public_card) < (card.private_card==card.public_card);
    return private_card< card.private_card;
}

LeducNode::LeducNode(const int& player_, const int& infoset_, const int& player_num_)
                    : Node(player_, infoset_, player_num_) {
    private_cards = std::vector<int>(player_num+1, -1);
    public_card = -1;
    round = 0;
    actions[0] = std::vector<BetAction>();
    actions[1] = std::vector<BetAction>();
    pot = std::vector<int>(player_num+1, 1); // put 1 chip per player in the pot
    pot[0] = 0;
    is_fold = std::vector<bool>(player_num+1, false);
    utility = std::vector<double>(player_num+1, 0.0);
}

bool LeducNode::IsRoundTerminal(){
    bool is_terminal[player_num+1];
    memset(is_terminal, false, sizeof(is_terminal));
    for(int i=actions[round].size()-1; i>=0; i--){
        is_terminal[actions[round][i].player] = true;
        if(actions[round][i].action == BetAction::Raise){
            break;
        }
    }
    for(int i=1; i<=player_num; i++){
        if(!is_terminal[i] && !is_fold[i]){
            return false;
        }
    }
    return true;
}

std::string LeducNode::GetInfosetString(const int& player){
    std::string infoset_str = "";
    infoset_str += std::to_string(private_cards[player]);
    for(int k=0; k<2; k++){
        for(int i=0; i<actions[k].size(); i++){
            infoset_str += actions[k][i].action_char[actions[k][i].action];
        }
        if(k == 0) infoset_str += std::to_string(public_card);
    }
    return infoset_str;
}

std::vector<int> LeducNode::Winner(){
    /*
        -1: not terminal yet
    */
    int fold_num = 0, non_fold_player = -1;
    for(int i=1; i<=player_num; i++){
        if(is_fold[i]){
            fold_num++;
        } else{
            non_fold_player = i;
        }
    }
    if(fold_num == player_num-1){
        return {non_fold_player};
    }

    if(round == 1 && IsRoundTerminal()){
        std::vector<int> winners = {1};
        for(int i=2;i<=player_num;i++){
            if(!is_fold[i]){
                if(Card(private_cards[i], public_card) == Card(private_cards[winners[0]], public_card)){
                    winners.push_back(i);
                } else if(Card(private_cards[winners[0]], public_card) < Card(private_cards[i], public_card)){
                    winners = {i};
                }
            }
        }
        return winners;
    }
    return {-1};
}

double LeducNode::GetUtility(const int& player){
    return utility[player];
}

void Leduc::AddPrivateCardNode(const int &player, int card[]){
    /*
        The index of card is 0, 1, 2, ..., rank * copies - 1
        The rank of a card is idx / copies
    */
    if(player == player_num + 1){
        std::string card_str = "";
        for(int i=1;i<=player_num;i++){
            card_str += std::to_string(card[i] / copies);
        }
        if(card_map.find(card_str) == card_map.end()){
            leduc_nodes.push_back(LeducNode(1, -1, player_num));
            for(int i=1; i<=player_num; i++){
                leduc_nodes[leduc_nodes.size()-1].private_cards[i] = card[i] / copies;
            }
            leduc_nodes[leduc_nodes.size()-1].infoset = GetInfosetIdx(1, leduc_nodes[leduc_nodes.size()-1].GetInfosetString(1));
            leduc_nodes[0].next_node.push_back(leduc_nodes.size()-1);
            leduc_nodes[0].chance.push_back(1.0);
            card_map[card_str] = leduc_nodes[0].chance.size-1;
        } else{
            leduc_nodes[0].chance[card_map[card_str]] += 1.0;
        }
        return;
    }
    for(int i=0;i<rank * copies;i++){
        bool flg = true;
        for(int j=1;j<player;j++){
            if(card[j] == i){
                flg = false;
                break;
            }
        }
        if(flg){
            card[player] = i;
            AddPrivateCardNode(player+1, card);
        }
    }
}

int Leduc::GetInfosetIdx(const int& player, const std::string& infoset_str){
    int infoset = (infoset_idx[player].find(infoset_str) == infoset_idx[player].end()) ? infoset_idx[player].size() : infoset_idx[player][infoset_str];
    infoset_idx[player][infoset_str] = infoset;
    return infoset;
}

void Leduc::AddRoundNode(const int& idx){
    if(leduc_nodes[idx].IsRoundTerminal()){
        auto winners = leduc_nodes[idx].Winner();
        if(winners[0] != -1){ // The game terminates
            std::vector<bool> is_winner(player_num+1, false);
            for(int i=0;i<winners.size();i++) is_winner[winners[i]] = true;
            int sum = 0.0;
            for(int i=1; i<=player_num; i++) sum += leduc_nodes[idx].pot[i];

            for(int player=1; player<=player_num; player++){
                leduc_nodes[idx].utility[player] = is_winner[player] ? sum / winners.size() : 0.0;
                leduc_nodes[idx].utility[player] -= leduc_nodes[idx].pot[player];
                if(player == winners[0]) leduc_nodes[idx].utility[player] += sum % winners.size();
                // leftover chips are given to the winner with the smallest index
            }
            /*for(int i=1, cur_idx=idx, player=leduc_nodes[idx].player; i<=player_num; i++){
                leduc_nodes[cur_idx].utility = is_winner[player] ? sum / winners.size() : 0.0;
                leduc_nodes[cur_idx].utility -= leduc_nodes[idx].pot[player];
                if(player == winners[0]) leduc_nodes[cur_idx].utility += sum % winners.size();
                // leftover chips are given to the winner with the smallest index

                if(i < player_num){
                    player = (player == player_num) ? 1 : player+1;
                    int infoset = GetInfosetIdx(player, leduc_nodes[idx].GetInfosetString(player));

                    leduc_nodes.push_back(LeducNode(player, infoset, player_num));
                    leduc_nodes[cur_idx].next_node.push_back(leduc_nodes.size()-1);
                    cur_idx = leduc_nodes.size()-1;
                }
            }*/
            return;
        } else{
            leduc_nodes[idx].round++;
            leduc_nodes[idx].actions[1].clear();
            leduc_nodes[idx].player = 0;
            std::vector<int> card_left(rank, copies);
            for(int i=1;i<=player_num;i++){
                card_left[leduc_nodes[idx].private_cards[i]]--;
            }
            for(int i=0; i<rank; i++) if(card_left[i] > 0){
                LeducNode new_node = leduc_nodes[idx];
                new_node.next_node.clear();
                new_node.public_card = i;

                for(int j=1;j<=player_num;j++) if(!leduc_nodes[idx].is_fold[j]){
                    new_node.player = j; // The player who will act next
                    new_node.infoset = GetInfosetIdx(j, new_node.GetInfosetString(j));
                    break;
                }
                leduc_nodes.push_back(new_node);
                leduc_nodes[idx].next_node.push_back(leduc_nodes.size()-1);
                leduc_nodes[idx].chance.push_back(double(card_left[i]));
                AddRoundNode(leduc_nodes.size()-1);
            }
            leduc_nodes[idx].chance = leduc_nodes[idx].chance / leduc_nodes[idx].chance.Sum();
        }
        return;
    }
    int max_pot = 1, next_player = -1, num_bet=0, last_bet_player=-1;
    for(int i=1;i<=player_num;i++) {
        max_pot = std::max(max_pot, leduc_nodes[idx].pot[i]);
        int player = (leduc_nodes[idx].player + i - 1) % player_num + 1;
        next_player = (!leduc_nodes[idx].is_fold[player] && next_player==-1) ? player : next_player;
    }
    for(int i=0; i<leduc_nodes[idx].actions[leduc_nodes[idx].round].size(); i++) if(leduc_nodes[idx].actions[leduc_nodes[idx].round][i].action == BetAction::Raise){
        num_bet++;
        last_bet_player = leduc_nodes[idx].actions[leduc_nodes[idx].round][i].player;
    }
    for(int a=0; a<3; a++) if(a!=BetAction::Raise || (num_bet<2 && last_bet_player != leduc_nodes[idx].player)) { // At most raise twice per round
        if(max_pot == leduc_nodes[idx].pot[leduc_nodes[idx].player] && a == BetAction::Fold) continue; // Cannot fold if the last player checked
        LeducNode new_node = leduc_nodes[idx];
        new_node.next_node.clear();
        new_node.actions[leduc_nodes[idx].round].push_back(BetAction(leduc_nodes[idx].player, a));
        if(a == BetAction::Fold){
            new_node.is_fold[leduc_nodes[idx].player] = true;
        } else if(a == BetAction::Call){
            new_node.pot[leduc_nodes[idx].player] = max_pot;
        } else if(a == BetAction::Raise){
            new_node.pot[leduc_nodes[idx].player] = max_pot + raise_amount[leduc_nodes[idx].round];
        }

        if(new_node.IsRoundTerminal() && new_node.Winner()[0] == -1){ // First round end but game not end
            new_node.player = 0;
            new_node.infoset = 2;
        } else{
            new_node.player = next_player;
            new_node.infoset = GetInfosetIdx(next_player, new_node.GetInfosetString(next_player));
        }

        leduc_nodes.push_back(new_node);
        leduc_nodes[idx].next_node.push_back(leduc_nodes.size()-1);
        AddRoundNode(leduc_nodes.size()-1);
    }
}

Leduc::Leduc(const int& player_num_, const int& rank_, const int& copies, const std::string& traverse_)
            : Environment(player_num_, traverse_), rank{rank_}, copies{copies} {
    
    for(int i=0;i<=player_num;i++) {
        infoset_idx.push_back(std::unordered_map<std::string, int>());
        infoset_idx[i]["Root"] = 0;
    }
    
    leduc_nodes.push_back(LeducNode(0, 1, player_num)); // sample private cards
    int card[player_num+1];
    memset(card, 0, sizeof(card));
    AddPrivateCardNode(1, card);
    leduc_nodes[0].chance = leduc_nodes[0].chance / leduc_nodes[0].chance.Sum();
    
    for(int i=1, upperbound=leduc_nodes.size(); i<upperbound; i++){
        AddRoundNode(i);
    }

    for(int i=0; i<leduc_nodes.size(); ++i){
        nodes.push_back(&leduc_nodes[i]);
    }
}
