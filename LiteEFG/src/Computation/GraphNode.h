#ifndef GRAPHNODE_H_
#define GRAPHNODE_H_

#include "Computation/Operations.h"
#include "Data/Vector.h"
#include "Computation/Projection.h"

#include <string>
#include <vector>
#include <variant>

using ObjectDoubleInt = std::variant<double, int>;

class GraphNode {
public:
    enum NodeIdx{
        utility = 0,
        action_set_size = 1,
        reach_prob = 2,
        opponent_reach_prob = 3,
        subtree_size = 4,
        start = 5
    };
    enum NodeStatus{
        smallest_status = 0, // smallest status
        static_backward_node = 0, // constant
        static_forward_node = 1, // constant
        backward_node = 2, // computed during backward pass
        forward_node = 3, // computed during forward pass
        status_num = 4,
    };

    using Object = std::variant<int, double, GraphNode>;

    static int num_nodes;
    static int graph_status, graph_color;
    static std::vector<GraphNode>* graph_nodes;
    static std::vector<std::pair<double, int> > constants_list;

    int idx, order; // idx: address of node, order: the order of node to be executed
    int status, color; //status and color of a node
    std::vector<int> dependency; // idx of the nodes that this node depends on
    std::shared_ptr<Operation> operation; // operation.Execute(dependency) --> *this
    
    GraphNode();
    GraphNode(const int& idx_, const std::vector<int>& dependency_, std::shared_ptr<Operation> operation_, const int& status_);
    GraphNode(const int& idx_, const std::initializer_list<int>& dependency_, std::shared_ptr<Operation> operation_, const int& status_);
    GraphNode(const double& val);
    GraphNode(const Vector& val);

    GraphNode AddConstScalar(const double& val);

    //template <typename T> GraphNode SingleVariableOperation(const T& operation=T()) const;

    template <typename T> GraphNode TwoVariableOperation(const Object& rhs, const T& operation=T()) const;

    void Inplace(const GraphNode& node);

    GraphNode operator+(const Object& rhs) const;
    GraphNode operator-(const Object& rhs) const;
    GraphNode operator-() const;
    GraphNode operator*(const Object& rhs) const;
    GraphNode operator/(const Object& rhs) const;

    GraphNode operator>(const Object& rhs) const;
    GraphNode operator<(const Object& rhs) const;
    GraphNode operator>=(const Object& rhs) const;
    GraphNode operator<=(const Object& rhs) const;
    GraphNode operator==(const Object& rhs) const;
    
    friend GraphNode operator+(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator-(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator*(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator/(const ObjectDoubleInt& lhs, const GraphNode& rhs);

    friend GraphNode operator>(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator<(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator>=(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator<=(const ObjectDoubleInt& lhs, const GraphNode& rhs);
    friend GraphNode operator==(const ObjectDoubleInt& lhs, const GraphNode& rhs);

    GraphNode Sum();
    GraphNode Mean();
    GraphNode Max();
    GraphNode Min();
    GraphNode Copy();
    
    GraphNode Exp();
    GraphNode Log();
    GraphNode Argmax();
    GraphNode Argmin();

    GraphNode Euclidean();
    GraphNode NegativeEntropy(const bool& shifted=false);
    GraphNode Normalize(const double& p_norm_, const bool& ignore_negative_);

    GraphNode Dot(const GraphNode& rhs);

    GraphNode Project(const std::string& distance_name, const Object& gamma);
    GraphNode Project(const std::string& distance_name, const Object& gamma, const GraphNode& mu);

    template <typename T> static GraphNode SingleVariableOperation(const GraphNode& node, const T& operation=T());
    template <typename T> static GraphNode TwoVariableOperation(const ObjectDoubleInt& lhs, const GraphNode& rhs, const T& operation=T());

    static GraphNode ConstVector(const Object& size, const Object& val);

    static GraphNode Sum(const GraphNode& node);
    static GraphNode Mean(const GraphNode& node);
    static GraphNode Max(const GraphNode& node);
    static GraphNode Min(const GraphNode& node);
    static GraphNode Copy(const GraphNode& node);

    static GraphNode Exp(const GraphNode& node);
    static GraphNode Log(const GraphNode& node);
    static GraphNode Argmax(const GraphNode& node);
    static GraphNode Argmin(const GraphNode& node);

    static GraphNode Euclidean(const GraphNode& node);
    static GraphNode NegativeEntropy(const GraphNode& node, const bool& shifted);
    static GraphNode Normalize(const GraphNode& node, const double& p_norm_, const bool& ignore_negative_);

    static GraphNode Dot(const GraphNode& lhs, const GraphNode& rhs);

    static GraphNode Maximum(const GraphNode& lhs, const Object& rhs);
    static GraphNode Minimum(const GraphNode& lhs, const Object& rhs);
    static GraphNode Aggregate(const GraphNode& node, const std::string& aggregator_name, const std::string& object_name="children", const std::string& player_name="self", const double& padding=0.0);

    static GraphNode Project(const GraphNode& node, const std::string& distance_name, const Object& gamma);
    static GraphNode Project(const GraphNode& node, const std::string& distance_name, const Object& gamma, const GraphNode& mu);
    
    static GraphNode Pow(const GraphNode& lhs, const Object& rhs);
    static GraphNode Pow(const ObjectDoubleInt& lhs, const GraphNode& rhs);
};

#endif