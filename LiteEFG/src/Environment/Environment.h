#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Data/Vector.h"
#include "Computation/Graph.h"
#include "Environment/Infoset.h"
#include "Environment/SequenceForm.h"

#include <vector>
#include <map>

class Environment{
public:
    enum Traverse{
        Enumerate = 0,
        Outcome = 1,
        External = 2
    };
    int player_num;
    std::vector<Node*> nodes, traverse_order;
    // nodes should always be in the same order as the game tree. i.e. parent should always be before children
    std::vector<std::vector<Infoset>> infosets;
    std::vector<Infoset*> traverse_infoset;
    std::vector<SequenceForm> sequence_form_strategies;
    std::vector<std::vector<std::string>> infoset_names;

    bool Flags_Initialized = false, Is_Aggregate_Opponents = false;
    int traverse;

    Graph graph;
    std::map<int, int> color_mapping;
    int num_colors;
    std::vector<bool> is_color_to_update;

    Environment(const int& player_num_, const std::string& traverse_="Enumerate");

    void SetGraph(const Graph& graph_);

    virtual void Initialize();
    double GetProb(Node* node, const int& strategy_node_idx, const int& action);
    Vector* GetProb(Node* node, const int& strategy_node_idx);

    void AggregateInformation(Infoset& infoset, const bool& is_parent, const int& node_status);
    void UpdateTraverse(const int& upd_player);
    void Update(const GraphNode& strategy_node, const int& upd_player=-1, std::vector<int> upd_color={-1});
    void Update(std::vector<GraphNode> strategy_nodes, const int& upd_player=-1, std::vector<int> upd_color={-1});
    
    void UpdateStrategy(const GraphNode& strategy_node, const bool& update_best=false);
    void UpdateStrategy(const std::vector<GraphNode>& strategy_node, const bool& update_best=false);
    
    std::vector<double> GetSequenceFormStrategy(const int& player, const GraphNode& strategy_node);

    //double Exploitability(const std::vector<SequenceForm>& sequence_form_strategies);
    void GetGradient(const std::vector<GraphNode>& strategy_node, const std::string& type_name="default");
    
    std::vector<double> Utility(const GraphNode& strategy_node, const std::string& type_name="default");
    std::vector<double> Utility(const std::vector<GraphNode>& strategy_node, const std::string& type_name="default");

    std::vector<double> Exploitability(const GraphNode& strategy_node, const std::string& type_name="default");
    std::vector<double> Exploitability(const std::vector<GraphNode>& strategy_node, const std::string& type_name="default");

    std::vector<std::pair<std::string, std::vector<double>> > GetValue(const int& player, const GraphNode& node);
    std::vector<std::pair<std::string, std::vector<double>> > GetStrategy(const int& player, const GraphNode& strategy_node, const std::string& type_name="default");
};

#endif