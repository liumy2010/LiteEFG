#ifndef SEQUENCEFORM_H_
#define SEQUENCEFORM_H_

#include "Environment/Infoset.h"
#include "Computation/GraphNode.h"

#include <vector>
#include <string>
#include <unordered_map>

class HistoryVersionStrategy{
public:
    Vector last_iterate, best_iterate, avg_iterate, linear_avg_iterate;
    double best_exploitability, timestep;

    HistoryVersionStrategy();
    HistoryVersionStrategy(const int& n);
    void UpdateStrategy(const Vector& strategy, const double& exploitability);
};

class SequenceForm{
public:
    std::vector<Infoset>* infosets;
    std::vector<int> start_sequence, end_sequence;
    Vector strategy, gradient, counterfactual_value;
    std::vector<HistoryVersionStrategy> history_version_strategies;
    std::vector<int> strategy_idx_map;

    SequenceForm(std::vector<Infoset>* infosets_);

    int GetIdx(const int& infoset, const int& action);
    bool IsSequenceForm(const Vector& input_strategy);
    bool IsSequenceForm();
    void UpdateStrategy(const int& strategy_node_idx, const double& exploitability);
    void GetSequenceFormStrategy(const int& strategy_node_idx, const std::string& type_name="default");
    //std::vector<double> GetStrategy();

    double GetUtility();
    double GetExploitability();
};

#endif