#include "Environment/SequenceForm.h"

#include "Basic/Constants.h"

#include <cmath>
#include <stdexcept>

HistoryVersionStrategy::HistoryVersionStrategy(){
    HistoryVersionStrategy(0);
}

HistoryVersionStrategy::HistoryVersionStrategy(const int& n){
    last_iterate = Vector(n, 0.0);
    best_iterate = Vector(n, 0.0);
    avg_iterate = Vector(n, 0.0);
    linear_avg_iterate = Vector(n, 0.0);
    best_exploitability = Constants::INF;
    timestep = 0.0;
}

void HistoryVersionStrategy::UpdateStrategy(const Vector& strategy, const double& exploitability){
    if(exploitability < best_exploitability){
        for(int i=0; i<strategy.size; i++){
            best_iterate[i] = strategy[i];
        }
        best_exploitability = exploitability;
    }
    for(int i=0; i<strategy.size; i++){
        last_iterate[i] = strategy[i];
        avg_iterate[i] = (avg_iterate[i] * (timestep / (timestep + 1.0))) + (strategy[i] / (timestep + 1.0));
        linear_avg_iterate[i] = (linear_avg_iterate[i] * (timestep / (timestep + 2.0))) + (strategy[i] * (2.0 / (timestep + 2.0)));
    }
    timestep += 1.0;
}

SequenceForm::SequenceForm(std::vector<Infoset>* infosets_) : infosets{infosets_} {
    start_sequence.resize(infosets->size());
    end_sequence.resize(infosets->size());
    strategy_idx_map.clear();

    int n = 0;
    for(int i = 0; i < infosets->size(); ++i){
        start_sequence[i] = n;
        n += (*infosets)[i].children.size();
        end_sequence[i] = n;
    }
    strategy = Vector(n, 0.0);
    gradient = Vector(n, 0.0);
    counterfactual_value = Vector(n, 0.0);
    history_version_strategies.clear();
}

int SequenceForm::GetIdx(const int& infoset, const int& action){
    return start_sequence[infoset] + action;
}

bool SequenceForm::IsSequenceForm(const Vector& input_strategy){
    if(input_strategy.size != strategy.size){
        throw std::invalid_argument("Strategy size does not match the sequence-form strategy size");
    }
    if(fabs(input_strategy[0] - 1.0) > Constants::EPS)
        throw std::invalid_argument("The reach probability of root is not 1");
    
    for(int i = 1; i < infosets -> size(); ++i){
        double sum = 0.0;
        for(int j = start_sequence[i]; j < end_sequence[i]; ++j){
            if(input_strategy[j] < 0.0){
                throw std::invalid_argument("Negative element in the sequence-form strategy");
            }
            sum += input_strategy[j];
        }
        int idx = GetIdx((*infosets)[i].parent.first, (*infosets)[i].parent.second);
        if(fabs(sum / std::max(Constants::EPS, input_strategy[idx]) - 1.0) > Constants::EPS && fabs(input_strategy[idx]) > Constants::EPS){
            throw std::invalid_argument("Invalid sequence-form strategy sum");
        }
    }
    return true;
}

bool SequenceForm::IsSequenceForm(){
    return IsSequenceForm(strategy);
}

void SequenceForm::UpdateStrategy(const int& strategy_node_idx, const double& exploitability){
    if(strategy_idx_map.size() <= strategy_node_idx)
        strategy_idx_map.resize(strategy_node_idx + 1, -1);
    if(strategy_idx_map[strategy_node_idx] == -1){
        strategy_idx_map[strategy_node_idx] = history_version_strategies.size();
        history_version_strategies.push_back(HistoryVersionStrategy(strategy.size));
    }
    history_version_strategies[strategy_idx_map[strategy_node_idx]].UpdateStrategy(strategy, exploitability);
}

//std::vector<double> SequenceForm::GetStrategy(){
//    return strategy.elements;
//}

void SequenceForm::GetSequenceFormStrategy(const int& strategy_node_idx, const std::string& type_name){
    (*infosets)[0].reach = 1.0;
    gradient.Set(0.0); //reset gradient
    if(strategy_idx_map.size() <= strategy_node_idx)
        strategy_idx_map.resize(strategy_node_idx + 1, -1);
    if(type_name != "default" && strategy_idx_map[strategy_node_idx] == -1)
        throw std::invalid_argument("Strategy name not found, please UpdateStrategy(strategy_name) first.");
    int idx = strategy_idx_map[strategy_node_idx];
    if(type_name == "last-iterate"){
        for(int i=0; i<strategy.size; i++){
            strategy[i] = history_version_strategies[idx].last_iterate[i];
        }
        return;
    }
    else if(type_name == "best-iterate"){
        if(std::fabs(history_version_strategies[idx].best_exploitability - Constants::INF) < Constants::EPS)
            throw std::invalid_argument("Best strategy not found, please UpdateStrategy(strategy_name, update_best=True) first.");
        for(int i=0; i<strategy.size; i++){
            strategy[i] = history_version_strategies[idx].best_iterate[i];
        }
        return;
    }
    else if(type_name == "avg-iterate"){
        for(int i=0; i<strategy.size; i++){
            strategy[i] = history_version_strategies[idx].avg_iterate[i];
        }
        return;
    }
    else if(type_name == "linear-avg-iterate"){
        for(int i=0; i<strategy.size; i++){
            strategy[i] = history_version_strategies[idx].linear_avg_iterate[i];
        }
        return;
    }
    else if(type_name != "default")
        throw std::invalid_argument("Invalid type. Must be: {last-iterate, best-iterate, avg-iterate, linear-avg-iterate}");

    for(int i=1; i<infosets->size(); ++i){
        Infoset& infoset = (*infosets)[i];
        infoset.reach = (*infosets)[infoset.parent.first].reach;
        if(infoset.parent.first != 0){
            infoset.reach *= (*infosets)[infoset.parent.first].results[strategy_node_idx][0][infoset.parent.second];
        }
        
        for(int j=start_sequence[i]; j<end_sequence[i]; ++j){
            strategy[j] = infoset.results[strategy_node_idx][0][j-start_sequence[i]] * infoset.reach;
        }
    }
    strategy[0] = 1.0;
}

double SequenceForm::GetUtility(){
    return strategy.Dot(gradient);
}

double SequenceForm::GetExploitability(){
    counterfactual_value = gradient;
    for(int i=infosets->size()-1; i>=1; --i){
        Infoset& infoset = (*infosets)[i];
        double ev = - Constants::INF;
        for(int j=start_sequence[i]; j<end_sequence[i]; ++j){
            ev = std::max(ev, counterfactual_value[j]);
        }
        counterfactual_value[GetIdx(infoset.parent.first, infoset.parent.second)] += ev;
    }

    return counterfactual_value[0] - GetUtility();
}