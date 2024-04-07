#ifndef INFOSET_H_
#define INFOSET_H_

#include "Basic/Constants.h"
#include "Data/Vector.h"
#include "Computation/Graph.h"
#include "Computation/GraphNode.h"

#include <vector>
#include <string>

class Node{
public:
    int player, infoset, idx, player_num;
    std::vector<int> next_node; // Next nodes in the game tree given the action
    // During construction of game environment, only need to specify player, infoset, next_node, and utility
    // Other variables will be handled by Environment

    std::pair<int, int> parent; // (Node, Action) pair
    std::vector<std::pair<int, int>> parent_infoset; // parent infoset of each player

    Vector reach; // Reach probability of each player
    Vector chance; // Chance probability if this is a chance node
    bool is_terminal;

    Node(const int& player_, const int& infoset_, const int& player_num_);
    
    static void Preprocessing(std::vector<Node*>& nodes, const int& player_num_);
    virtual double GetUtility(const int& player);
};

class Infoset{
public:
    /*
        graph: computation graph given by the user
    */
    Graph graph;
    std::vector<std::vector<int>> children; // children infoset of each sequence (infoset, action)
    std::pair<int, int> parent; // parent sequence (infoset, action) of each infoset
    int first_visited, player, size;
    double reach;

    std::vector<std::vector<Vector> > results; // results of computation graph
    std::vector<int> aggregator_dependency; // dependency of the aggregator
    std::vector<int> aggregator_type; // type of the aggregator

    Infoset();

    void UpdateGraph(const int& status=GraphNode::NodeStatus::backward_node);
    void InitializeAggregator();
    void AggregateChildren(Infoset& child, const int& status=GraphNode::NodeStatus::backward_node);
    void AggregateParent(Infoset& parent, const int& status=GraphNode::NodeStatus::backward_node);
    void InitializeGraph(const double& reach);

    Vector GetResult(const int& opIndex);
}; // Idx 0 is the root infoset

#endif