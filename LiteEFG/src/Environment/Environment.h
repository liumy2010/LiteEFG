#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Data/Vector.h"
#include "Computation/Graph.h"
#include "Environment/Infoset.h"
#include "Environment/SequenceForm.h"

#include <vector>
#include <map>
#include <array>

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
    int num_colors = 0;
    std::vector<bool> is_color_to_update;

    Environment(const int& player_num_, const std::string& traverse_="Enumerate");

    void SetGraph(const Graph& graph_);
    void ResetParallelSchedule();

    virtual void Initialize();
    double GetProb(Node* node, const int& strategy_node_idx, const int& action);
    Vector* GetProb(Node* node, const int& strategy_node_idx);

    void AggregateInformation(Infoset& infoset, const bool& is_parent, const int& node_status);
    void UpdateTraverse(const int& upd_player, const int& current_traverse);
    void Update(const GraphNode& strategy_node, const int& upd_player=-1, std::vector<int> upd_color={-1}, const std::string& traverse_type="default");
    void Update(std::vector<GraphNode> strategy_nodes, const int& upd_player=-1, std::vector<int> upd_color={-1}, const std::string& traverse_type="default");
    
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

    void SetValue(const int& player, const GraphNode& node, const std::vector<std::vector<double>>& values);
    void SetValue(const int& player, const GraphNode& node, const std::vector<double>& values);

    virtual ~Environment();

private:
    struct ParallelInfoset {
        Infoset* infoset;
        int depth = 0;
        std::vector<std::pair<Infoset*, int>> children;
        std::array<double, 2> work{};
    };
    struct ParallelLayer {
        std::vector<int> infosets;
        // Separate weighted contiguous partitions for backward and forward.
        std::array<std::vector<size_t>, 2> boundaries;
    };
    std::vector<ParallelInfoset> parallel_infosets;
    std::vector<ParallelLayer> parallel_layers;
    std::array<std::vector<int>, 2> parallel_random_nodes;
    std::array<bool, 2> parallel_safe{{true, true}};
    int scheduled_threads = 0;
    void PrepareParallelSchedule();
    bool UpdateParallel(const int& status, const int& current_traverse);
    void GatherChildren(ParallelInfoset& entry, int status, bool sampled);
};

#endif
