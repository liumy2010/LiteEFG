#ifndef NFG_H_
#define NFG_H_

#include "Data/Vector.h"
#include "Data/Tensor.h"
#include "Environment/Infoset.h"
#include "Environment/Environment.h"

#include <vector>

class NFGNode : public Node{
public:
    std::vector<int> actions;
    std::vector<double> utility;
    bool is_terminal;

    NFGNode(const int& player_, const int& infoset_, const int& player_num_);
    double GetUtility(const int& player) override;
};

class NFG : public Environment{
public:
    std::vector<Tensor> utility_matrices;
    std::vector<NFGNode> nfg_nodes;

    NFG(const int& player_num_, const std::vector<Tensor>& utility_matrix, const std::string& traverse_="Enumerate");
    void Initialize();
};

#endif