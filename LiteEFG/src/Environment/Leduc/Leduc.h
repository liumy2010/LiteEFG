#ifndef LEDUC_H_
#define LEDUC_H_

#include "Data/Vector.h"
#include "Data/Tensor.h"
#include "Environment/Infoset.h"
#include "Environment/Environment.h"

#include <vector>
#include <unordered_map>

class Card{
public:
    int private_card, public_card;
    Card(const int& private_card_, const int& public_card_) : private_card(private_card_), public_card(public_card_){};
    bool operator==(const Card& card);
    bool operator<(const Card& card);
};

class BetAction{
public:
    enum ActionType{
        Fold = 0,
        Call = 1,
        Raise = 2
    };
    int player, action;
    BetAction(const int& player_, const int& action_) : player{player_}, action{action_} {};
    const char action_char[3] = {'F', 'C', 'R'};
};

class LeducNode : public Node{
public:
    std::vector<int> private_cards;
    int public_card;
    int round;
    std::vector<BetAction> actions[2];
    std::vector<int> pot;
    std::vector<bool> is_fold;
    std::vector<double> utility;

    LeducNode(const int& player_, const int& infoset_, const int& player_num_);
    bool IsRoundTerminal();
    std::vector<int> Winner();
    std::string GetInfosetString(const int& player);
    double GetUtility(const int& player) override;
};

class Leduc : public Environment{
public:
    std::vector<LeducNode> leduc_nodes;
    const int raise_amount[2] = {1, 2};
    std::unordered_map<std::string, int> card_map;
    std::vector<std::unordered_map<std::string, int>> infoset_idx;
    int rank, copies;
    
    int GetInfosetIdx(const int& player, const std::string& infoset_str);
    void AddPrivateCardNode(const int &player, int card[]);
    void AddRoundNode(const int &idx);
    Leduc(const int& player_num_=2, const int& rank_=3, const int& copies=2, const std::string& traverse_="Enumerate");
};

#endif