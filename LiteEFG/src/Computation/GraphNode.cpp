#include "Computation/GraphNode.h"

#include "Basic/Constants.h"
#include "Computation/Operations.h"
#include "Computation/Static.h"
#include "Computation/Projection.h"

#include <cmath>
#include <stdexcept>

int GraphNode::num_nodes = 5;
int GraphNode::graph_status = GraphNode::NodeStatus::smallest_status;
int GraphNode::graph_color = 0;
std::vector<GraphNode>* GraphNode::graph_nodes = NULL;
std::vector<std::pair<double, int> > GraphNode::constants_list;
std::shared_ptr<GraphBuilder> GraphNode::active_builder;

void GraphNode::SyncBuilder() {
    if(active_builder && active_builder->nodes) {
        active_builder->num_nodes = num_nodes;
        active_builder->constants = constants_list;
    }
}

GraphBuilderContext GraphNode::CaptureContext() {
    SyncBuilder();
    return {active_builder, graph_status, graph_color};
}

void GraphNode::Activate(const std::shared_ptr<GraphBuilder>& builder) {
    if(!builder || !builder->nodes)
        throw std::invalid_argument("Cannot activate a destroyed native graph");
    SyncBuilder();
    active_builder = builder;
    graph_nodes = builder->nodes;
    num_nodes = builder->num_nodes;
    constants_list = builder->constants;
    graph_status = NodeStatus::smallest_status;
    graph_color = 0;
}

void GraphNode::ClearContext() {
    SyncBuilder();
    active_builder.reset();
    graph_nodes = nullptr;
    num_nodes = NodeIdx::start;
    constants_list.clear();
    graph_status = NodeStatus::smallest_status;
    graph_color = 0;
}

void GraphNode::RestoreContext(const GraphBuilderContext& context) {
    if(context.owner && context.owner->nodes) Activate(context.owner);
    else ClearContext();
    graph_status = context.status;
    graph_color = context.color;
}

std::shared_ptr<GraphBuilder> GraphNode::RequireBuilder() {
    if(!active_builder || !active_builder->nodes || graph_nodes != active_builder->nodes)
        throw std::invalid_argument("No active native graph; create a LiteEFG.Graph before creating nodes");
    return active_builder;
}

std::shared_ptr<GraphBuilder> GraphNode::RequireOwner() const {
    const auto builder = owner.lock();
    if(!builder || !builder->nodes)
        throw std::invalid_argument("GraphNode has no live owning graph");
    if(idx < 0 || order < 0)
        throw std::invalid_argument("Cannot use an uninitialized GraphNode");
    return builder;
}

void GraphNode::CheckOwner(const GraphNode& other) const {
    if(RequireOwner() != other.RequireOwner())
        throw std::invalid_argument("Cannot combine GraphNodes from different graphs");
}

GraphBuilderScope::GraphBuilderScope(const GraphNode& node)
    : previous(GraphNode::CaptureContext()) {
    GraphNode::Activate(node.RequireOwner());
    // The dynamic forward/backward block applies to operations on existing
    // nodes, even when their owner is not the most recently created graph.
    GraphNode::graph_status = previous.status;
    GraphNode::graph_color = previous.color;
}

GraphBuilderScope::~GraphBuilderScope() {
    GraphNode::RestoreContext(previous);
}

GraphNode::GraphNode() : idx(-1), order(-1), status(GraphNode::NodeStatus::smallest_status), operation(NULL), color{GraphNode::graph_color}{dependency.clear();}

GraphNode::GraphNode(const int& idx_, const std::vector<int>& dependency_, std::shared_ptr<Operation> operation_, const int& status_)
                    : owner(active_builder), idx(idx_), order(idx_), dependency(dependency_), operation(operation_), status(status_), color{GraphNode::graph_color}{}

GraphNode::GraphNode(const int& idx_, const std::initializer_list<int>& dependency_, std::shared_ptr<Operation> operation_, const int& status_)
                    : owner(active_builder), idx(idx_), order(idx_), dependency(dependency_), operation(operation_), status(status_), color{GraphNode::graph_color}{}

GraphNode::GraphNode(const double& val){
    owner = RequireBuilder();
    constants_list.push_back({val, num_nodes});
    idx = order = num_nodes++;
    dependency.clear();
    operation = std::make_shared<StaticConstVector>(StaticConstVector(1, val));
    status = GraphNode::NodeStatus::smallest_status;
    color = GraphNode::graph_color;
}

GraphNode::GraphNode(const Vector& val) {
    owner = RequireBuilder();
    idx = order = num_nodes++;
    dependency.clear();
    operation = std::make_shared<StaticConstVector>(StaticConstVector(val));
    status = GraphNode::NodeStatus::smallest_status;
    color = GraphNode::graph_color;
}

GraphNode GraphNode::AddConstScalar(const double& val){
    RequireBuilder();
    for(auto& elem : constants_list){
        if(fabs(elem.first - val) < Constants::EPS){
            return (*graph_nodes)[elem.second];
        }
    }
    return GraphNode(val);
}

void GraphNode::Inplace(const GraphNode& node){
    CheckOwner(node);
    GraphBuilderScope scope(*this);
    if(node.order >= graph_nodes->size())
        throw std::invalid_argument("Invalid inplace node order");
    if((*graph_nodes)[node.order].order != node.order){
        throw std::invalid_argument("Unexpected error");
    }
    (*graph_nodes)[node.order].idx = idx; // change the address to store
}

template <typename T> GraphNode GraphNode::SingleVariableOperation(const GraphNode& node, const T& operation) {
    GraphBuilderScope scope(node);
    GraphNode::graph_nodes->push_back(GraphNode(GraphNode::num_nodes++, {node.idx}, std::make_shared<T>(operation), GraphNode::graph_status));
    return GraphNode::graph_nodes->back();
}

template <typename T> GraphNode GraphNode::TwoVariableOperation(const Object& rhs, const T& operation) const {
    if(std::holds_alternative<GraphNode>(rhs)) CheckOwner(std::get<GraphNode>(rhs));
    GraphBuilderScope scope(*this);
    if(std::holds_alternative<double>(rhs)){
        graph_nodes->push_back(GraphNode(std::get<double>(rhs)));
        graph_nodes->push_back(GraphNode(num_nodes++, {idx, graph_nodes->back().idx}, std::make_shared<T>(operation), graph_status));
    } else if(std::holds_alternative<int>(rhs)){
        graph_nodes->push_back(GraphNode(std::get<int>(rhs)));
        graph_nodes->push_back(GraphNode(num_nodes++, {idx, graph_nodes->back().idx}, std::make_shared<T>(operation), graph_status));
    } else {
        graph_nodes->push_back(GraphNode(num_nodes++, {idx, std::get<GraphNode>(rhs).idx}, std::make_shared<T>(operation), graph_status));
    }
    return graph_nodes->back();
}

template <typename T> GraphNode GraphNode::TwoVariableOperation(const ObjectDoubleInt& lhs, const GraphNode& rhs, const T& operation) {
    GraphBuilderScope scope(rhs);
    if(std::holds_alternative<double>(lhs)){
        GraphNode::graph_nodes->push_back(GraphNode(std::get<double>(lhs)));
        GraphNode::graph_nodes->push_back(GraphNode(GraphNode::num_nodes++, {GraphNode::graph_nodes->back().idx, rhs.idx}, std::make_shared<T>(operation), GraphNode::graph_status));
    } else{
        GraphNode::graph_nodes->push_back(GraphNode(std::get<int>(lhs)));
        GraphNode::graph_nodes->push_back(GraphNode(GraphNode::num_nodes++, {GraphNode::graph_nodes->back().idx, rhs.idx}, std::make_shared<T>(operation), GraphNode::graph_status));
    }
    return GraphNode::graph_nodes->back();
}

GraphNode GraphNode::ConstVector(const Object& size, const Object& val){
    if(std::holds_alternative<double>(size) ||
       (std::holds_alternative<int>(size) && std::get<int>(size) < 0))
        throw std::invalid_argument("Invalid const size: expected a nonnegative int or GraphNode");
    if(!std::holds_alternative<GraphNode>(size) && !std::holds_alternative<GraphNode>(val)) RequireBuilder();
    if(std::holds_alternative<int>(size) && std::holds_alternative<double>(val)){
        GraphNode::graph_nodes->push_back(GraphNode(Vector(std::get<int>(size), std::get<double>(val))));
        return GraphNode::graph_nodes->back();
    } else if(std::holds_alternative<int>(size) && std::holds_alternative<int>(val)){
        GraphNode::graph_nodes->push_back(GraphNode(Vector(std::get<int>(size), std::get<int>(val))));
        return GraphNode::graph_nodes->back();
    } else if(std::holds_alternative<int>(size) && std::holds_alternative<GraphNode>(val)){
        return TwoVariableOperation<StaticConstVector>(std::get<int>(size), std::get<GraphNode>(val));
    } else if(std::holds_alternative<GraphNode>(size)){
        return std::get<GraphNode>(size).TwoVariableOperation<StaticConstVector>(val);
    } else {
        throw std::invalid_argument("Invalid arguments for ConstVector: size must be either int or GraphNode, val must be int, double, or GraphNode");
    }
}

GraphNode GraphNode::operator+(const Object& rhs) const {
    return TwoVariableOperation<AddOperation>(rhs);
}

GraphNode GraphNode::operator-(const Object& rhs) const {
    return TwoVariableOperation<SubOperation>(rhs);
}

GraphNode GraphNode::operator-() const {
    return TwoVariableOperation<SubOperation>(0.0, *this);
}

GraphNode GraphNode::operator*(const Object& rhs) const {
    return TwoVariableOperation<MulOperation>(rhs);
}

GraphNode GraphNode::operator/(const Object& rhs) const {
    return TwoVariableOperation<DivOperation>(rhs);
}

GraphNode GraphNode::operator>(const Object& rhs) const {
    return TwoVariableOperation<CompareOperation>(rhs, CompareOperation(CompareOperation::ComparatorType::greater_than));
}

GraphNode GraphNode::operator<(const Object& rhs) const {
    return TwoVariableOperation<CompareOperation>(rhs, CompareOperation(CompareOperation::ComparatorType::less_than));
}

GraphNode GraphNode::operator>=(const Object& rhs) const {
    return TwoVariableOperation<CompareOperation>(rhs, CompareOperation(CompareOperation::ComparatorType::greater_than_or_equal));
}

GraphNode GraphNode::operator<=(const Object& rhs) const {
    return TwoVariableOperation<CompareOperation>(rhs, CompareOperation(CompareOperation::ComparatorType::less_than_or_equal));
}

GraphNode GraphNode::operator==(const Object& rhs) const {
    return TwoVariableOperation<CompareOperation>(rhs, CompareOperation(CompareOperation::ComparatorType::equal));
}

GraphNode operator+(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<AddOperation>(lhs, rhs);
}

GraphNode operator-(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<SubOperation>(lhs, rhs);
}

GraphNode operator*(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<MulOperation>(lhs, rhs);
}

GraphNode operator/(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<DivOperation>(lhs, rhs);
}

GraphNode operator>(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<CompareOperation>(lhs, rhs, CompareOperation(CompareOperation::ComparatorType::greater_than));
}

GraphNode operator<(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<CompareOperation>(lhs, rhs, CompareOperation(CompareOperation::ComparatorType::less_than));
}

GraphNode operator>=(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<CompareOperation>(lhs, rhs, CompareOperation(CompareOperation::ComparatorType::greater_than_or_equal));
}

GraphNode operator<=(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<CompareOperation>(lhs, rhs, CompareOperation(CompareOperation::ComparatorType::less_than_or_equal));
}

GraphNode operator==(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return GraphNode::TwoVariableOperation<CompareOperation>(lhs, rhs, CompareOperation(CompareOperation::ComparatorType::equal));
}

GraphNode GraphNode::Sum(){
    return GraphNode::SingleVariableOperation<SumOperation>(*this);
}

GraphNode GraphNode::Sum(const GraphNode& node){
    return SingleVariableOperation<SumOperation>(node);
}

GraphNode GraphNode::Mean(){
    return GraphNode::SingleVariableOperation<MeanOperation>(*this);
}

GraphNode GraphNode::Mean(const GraphNode& node){
    return SingleVariableOperation<MeanOperation>(node);
}

GraphNode GraphNode::Max(){
    return GraphNode::SingleVariableOperation<MaxOperation>(*this);
}

GraphNode GraphNode::Max(const GraphNode& node){
    return SingleVariableOperation<MaxOperation>(node);
}

GraphNode GraphNode::Min(){
    return GraphNode::SingleVariableOperation<MinOperation>(*this);
}

GraphNode GraphNode::Min(const GraphNode& node){
    return SingleVariableOperation<MinOperation>(node);
}

GraphNode GraphNode::Copy(){
    return GraphNode::SingleVariableOperation<CopyOperation>(*this);
}

GraphNode GraphNode::Copy(const GraphNode& node){
    return SingleVariableOperation<CopyOperation>(node);
}

GraphNode GraphNode::Exp(){
    return GraphNode::SingleVariableOperation<ExpOperation>(*this);
}

GraphNode GraphNode::Exp(const GraphNode& node){
    return SingleVariableOperation<ExpOperation>(node);
}

GraphNode GraphNode::Log(){
    return GraphNode::SingleVariableOperation<LogOperation>(*this);
}

GraphNode GraphNode::Log(const GraphNode& node){
    return SingleVariableOperation<LogOperation>(node);
}

GraphNode GraphNode::Argmax(){
    return GraphNode::SingleVariableOperation<ArgmaxOperation>(*this);
}

GraphNode GraphNode::Argmax(const GraphNode& node){
    return SingleVariableOperation<ArgmaxOperation>(node);
}

GraphNode GraphNode::Argmin(){
    return GraphNode::SingleVariableOperation<ArgminOperation>(*this);
}

GraphNode GraphNode::Argmin(const GraphNode& node){
    return SingleVariableOperation<ArgminOperation>(node);
}

GraphNode GraphNode::Euclidean(){
    return GraphNode::SingleVariableOperation<EuclideanOperation>(*this);
}

GraphNode GraphNode::Euclidean(const GraphNode& node){
    return SingleVariableOperation<EuclideanOperation>(node);
}

GraphNode GraphNode::NegativeEntropy(const bool& shifted){
    return GraphNode::SingleVariableOperation<NegativeEntropyOperation>(*this, NegativeEntropyOperation(shifted));
}

GraphNode GraphNode::NegativeEntropy(const GraphNode& node, const bool& shifted=false){
    return SingleVariableOperation<NegativeEntropyOperation>(node, NegativeEntropyOperation(shifted));
}

GraphNode GraphNode::Normalize(const double& p_norm_, const bool& ignore_negative_){
    return GraphNode::SingleVariableOperation<NormalizeOperation>(*this, NormalizeOperation(p_norm_, ignore_negative_));
}

GraphNode GraphNode::Normalize(const GraphNode& node, const double& p_norm_, const bool& ignore_negative_){
    return SingleVariableOperation<NormalizeOperation>(node, NormalizeOperation(p_norm_, ignore_negative_));
}

GraphNode GraphNode::Dot(const GraphNode& rhs){
    return TwoVariableOperation<DotOperation>(rhs);
}

GraphNode GraphNode::Dot(const GraphNode& lhs, const GraphNode& rhs){
    return lhs.TwoVariableOperation<DotOperation>(rhs);
}

GraphNode GraphNode::Maximum(const GraphNode& lhs, const Object& rhs){
    return lhs.TwoVariableOperation<MaximumOperation>(rhs);
}

GraphNode GraphNode::Minimum(const GraphNode& lhs, const Object& rhs){
    return lhs.TwoVariableOperation<MinimumOperation>(rhs);
}

GraphNode GraphNode::Aggregate(const GraphNode& node, const std::string& aggregator_name, const std::string& object_name, const std::string& player_name, const double& padding){
    std::shared_ptr<Operation> aggregator = NULL;
    if(aggregator_name == "sum") {
        aggregator = std::make_shared<SumOperation>(SumOperation());
    } else if(aggregator_name == "mean") {
        aggregator = std::make_shared<MeanOperation>(MeanOperation());
    } else if(aggregator_name == "max") {
        aggregator = std::make_shared<MaxOperation>(MaxOperation());
    } else if(aggregator_name == "min") {
        aggregator = std::make_shared<MinOperation>(MinOperation());
    } else {
        throw std::invalid_argument("Invalid aggregator name");
    }
    return SingleVariableOperation<AggregateOperation>(node, AggregateOperation(aggregator, object_name, player_name, padding));
}

GraphNode GraphNode::Project(const std::string& distance_name, const Object& gamma){
    return TwoVariableOperation<ProjectionOperation>(gamma, ProjectionOperation(distance_name));
}

GraphNode GraphNode::Project(const GraphNode& node, const std::string& distance_name, const Object& gamma){
    return node.TwoVariableOperation<ProjectionOperation>(gamma, ProjectionOperation(distance_name));
}

GraphNode GraphNode::Project(const std::string& distance_name, const Object& gamma, const GraphNode& mu){
    CheckOwner(mu);
    GraphBuilderScope scope(*this);
    TwoVariableOperation<ProjectionOperation>(gamma, ProjectionOperation(distance_name));
    graph_nodes->back().dependency.push_back(mu.idx);
    return graph_nodes->back();
}

GraphNode GraphNode::Project(const GraphNode& node, const std::string& distance_name, const Object& gamma, const GraphNode& mu){
    node.CheckOwner(mu);
    GraphBuilderScope scope(node);
    node.TwoVariableOperation<ProjectionOperation>(gamma, ProjectionOperation(distance_name));
    graph_nodes->back().dependency.push_back(mu.idx);
    return graph_nodes->back();
}

GraphNode GraphNode::Pow(const GraphNode& lhs, const Object& rhs){
    return lhs.TwoVariableOperation<PowOperation>(rhs);
}

GraphNode GraphNode::Pow(const ObjectDoubleInt& lhs, const GraphNode& rhs){
    return TwoVariableOperation<PowOperation>(lhs, rhs);
}

GraphNode GraphNode::Concat(const std::vector<GraphNode>& nodes){
    if(nodes.empty()) throw std::invalid_argument("cat requires at least one GraphNode");
    for(const auto& node : nodes) nodes.front().CheckOwner(node);
    GraphBuilderScope scope(nodes.front());
    std::vector<int> dependency;
    for(const auto& node : nodes){
        dependency.push_back(node.idx);
    }
    graph_nodes->push_back(GraphNode(num_nodes++, dependency, std::make_shared<ConcatOperation>(ConcatOperation()), graph_status));
    return graph_nodes->back();
}

GraphNode GraphNode::RandomUniform(const GraphNode& node, const double& lower, const double& upper){
    return SingleVariableOperation<RandomUniformOperation>(node, RandomUniformOperation(lower, upper));
}

GraphNode GraphNode::RandomNormal(const GraphNode& node, const double& mean, const double& stddev){
    return SingleVariableOperation<RandomNormalOperation>(node, RandomNormalOperation(mean, stddev));
}

GraphNode GraphNode::RandomExponential(const GraphNode& node, const double& lambda){
    return SingleVariableOperation<RandomExponentialOperation>(node, RandomExponentialOperation(lambda));
}
