#include "Computation/Checkpoint.h"

#include "Basic/BasicFunction.h"
#include "Computation/Projection.h"
#include "Computation/Static.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::vector<double> SaveVector(const Vector& vector) {
    std::vector<double> values(vector.size);
    for (int i = 0; i < vector.size; ++i) values[i] = vector[i];
    return values;
}

Vector LoadVector(py::handle values) {
    return Vector(py::cast<std::vector<double>>(values));
}

py::object SaveOperation(const std::shared_ptr<Operation>& operation) {
    if (!operation) return py::none();
    py::dict state;
    state["name"] = operation->name;
    state["is_static"] = operation->is_static;
    state["tmp"] = SaveVector(operation->tmp);
    state["info"] = SaveVector(operation->info);
    if (const auto* value = dynamic_cast<const StaticConstVector*>(operation.get())) {
        state["elements"] = SaveVector(value->elements);
    } else if (const auto* value = dynamic_cast<const NegativeEntropyOperation*>(operation.get())) {
        state["shifted"] = value->shifted;
    } else if (const auto* value = dynamic_cast<const NormalizeOperation*>(operation.get())) {
        state["p_norm"] = value->p_norm;
        state["ignore_negative"] = value->ignore_negative;
    } else if (const auto* value = dynamic_cast<const CompareOperation*>(operation.get())) {
        state["type"] = value->GetType();
    } else if (const auto* value = dynamic_cast<const AggregateOperation*>(operation.get())) {
        state["aggregator"] = SaveOperation(value->aggregator);
    } else if (const auto* value = dynamic_cast<const ProjectionOperation*>(operation.get())) {
        state["distance_name"] = value->distance_name;
        // Projection's lowerbound is scratch, assigned on every Execute call.
    } else if (const auto* value = dynamic_cast<const RandomUniformOperation*>(operation.get())) {
        state["lower"] = value->GetLower();
        state["upper"] = value->GetUpper();
    } else if (const auto* value = dynamic_cast<const RandomNormalOperation*>(operation.get())) {
        state["mean"] = value->GetMean();
        state["stddev"] = value->GetStddev();
    } else if (const auto* value = dynamic_cast<const RandomExponentialOperation*>(operation.get())) {
        state["lambda"] = value->GetLambda();
    }
    return state;
}

std::shared_ptr<Operation> LoadOperation(py::handle value) {
    if (value.is_none()) return nullptr;
    const auto state = py::cast<py::dict>(value);
    const auto name = state["name"].cast<std::string>();
    const bool is_static = state["is_static"].cast<bool>();
    std::shared_ptr<Operation> operation;

#define LOAD_SIMPLE_OPERATION(type) \
    if (name == #type) operation = std::make_shared<type##Operation>(is_static); else
    LOAD_SIMPLE_OPERATION(Copy)
    LOAD_SIMPLE_OPERATION(Add)
    LOAD_SIMPLE_OPERATION(Sub)
    LOAD_SIMPLE_OPERATION(Mul)
    LOAD_SIMPLE_OPERATION(Div)
    LOAD_SIMPLE_OPERATION(Exp)
    LOAD_SIMPLE_OPERATION(Log)
    LOAD_SIMPLE_OPERATION(Sum)
    LOAD_SIMPLE_OPERATION(Mean)
    LOAD_SIMPLE_OPERATION(Max)
    LOAD_SIMPLE_OPERATION(Min)
    LOAD_SIMPLE_OPERATION(Dot)
    LOAD_SIMPLE_OPERATION(Argmax)
    LOAD_SIMPLE_OPERATION(Argmin)
    LOAD_SIMPLE_OPERATION(Maximum)
    LOAD_SIMPLE_OPERATION(Minimum)
    LOAD_SIMPLE_OPERATION(Euclidean)
    LOAD_SIMPLE_OPERATION(Pow)
    LOAD_SIMPLE_OPERATION(Concat)
    if (name == "StaticConstVector") {
        operation = std::make_shared<StaticConstVector>(LoadVector(state["elements"]));
    } else if (name == "NegativeEntropy") {
        operation = std::make_shared<NegativeEntropyOperation>(state["shifted"].cast<bool>(), is_static);
    } else if (name == "Normalize") {
        operation = std::make_shared<NormalizeOperation>(state["p_norm"].cast<double>(),
            state["ignore_negative"].cast<bool>(), is_static);
    } else if (name == "Compare") {
        operation = std::make_shared<CompareOperation>(state["type"].cast<int>(), is_static);
    } else if (name == "Aggregate") {
        auto aggregator = LoadOperation(state["aggregator"]);
        if (!aggregator) throw std::invalid_argument("Checkpoint aggregate has no aggregator");
        const auto info = LoadVector(state["info"]);
        if (info.size != 3) throw std::invalid_argument("Invalid checkpoint aggregate information");
        operation = std::make_shared<AggregateOperation>(aggregator,
            info[AggregateOperation::object] > 0.0 ? "children" : "parent",
            info[AggregateOperation::player] > 0.0 ? "self" : "opponents",
            info[AggregateOperation::padding], is_static);
    } else if (name == "Projection") {
        operation = std::make_shared<ProjectionOperation>(state["distance_name"].cast<std::string>(), is_static);
    } else if (name == "RandomUniform") {
        operation = std::make_shared<RandomUniformOperation>(state["lower"].cast<double>(),
            state["upper"].cast<double>(), is_static);
    } else if (name == "RandomNormal") {
        operation = std::make_shared<RandomNormalOperation>(state["mean"].cast<double>(),
            state["stddev"].cast<double>(), is_static);
    } else if (name == "RandomExponential") {
        operation = std::make_shared<RandomExponentialOperation>(state["lambda"].cast<double>(), is_static);
    } else {
        throw std::invalid_argument("Unknown checkpoint operation: " + name);
    }
#undef LOAD_SIMPLE_OPERATION

    operation->is_static = is_static;
    operation->tmp = LoadVector(state["tmp"]);
    operation->info = LoadVector(state["info"]);
    return operation;
}

struct PreserveGraphBuilder {
    GraphBuilderContext context = GraphNode::CaptureContext();
    ~PreserveGraphBuilder() {
        GraphNode::RestoreContext(context);
    }
};

template <typename T> std::string SaveRandomComponent(const T& value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return stream.str();
}

template <typename T> T LoadRandomComponent(py::handle value) {
    std::istringstream stream(py::cast<std::string>(value));
    stream.imbue(std::locale::classic());
    T result;
    if (!(stream >> result)) throw std::invalid_argument("Invalid checkpoint random state");
    stream >> std::ws;
    if (!stream.eof()) throw std::invalid_argument("Trailing data in checkpoint random state");
    return result;
}

}

py::tuple Checkpoint::SaveNode(const GraphNode& node) {
    return py::make_tuple(node.idx, node.order, node.status, node.color,
        node.dependency, SaveOperation(node.operation));
}

GraphNode Checkpoint::LoadNode(const py::tuple& state) {
    if (state.size() != 6) throw std::invalid_argument("Invalid checkpoint graph node");
    GraphNode node;
    node.idx = state[0].cast<int>();
    node.order = state[1].cast<int>();
    node.status = state[2].cast<int>();
    node.color = state[3].cast<int>();
    node.dependency = state[4].cast<std::vector<int>>();
    node.operation = LoadOperation(state[5]);
    if (node.status < 0 || node.status >= GraphNode::status_num)
        throw std::invalid_argument("Invalid checkpoint node status");
    return node;
}

py::dict Checkpoint::SaveGraph(const Graph& graph) {
    py::dict state;
    state["version"] = 1;
    state["order"] = graph.order;
    state["timestep"] = graph.timestep;
    state["start_idx"] = std::vector<int>(std::begin(graph.start_idx), std::end(graph.start_idx));
    py::list nodes;
    for (const auto& node : graph.graph_nodes) nodes.append(SaveNode(node));
    state["nodes"] = nodes;
    state["utility"] = SaveNode(graph.utility);
    state["opponent_reach_prob"] = SaveNode(graph.opponent_reach_prob);
    state["reach_prob"] = SaveNode(graph.reach_prob);
    state["action_set_size"] = SaveNode(graph.action_set_size);
    state["subtree_size"] = SaveNode(graph.subtree_size);
    // inputs contains transient pointers rebound by Graph::Execute.
    return state;
}

Graph Checkpoint::LoadGraph(const py::dict& state) {
    if (state["version"].cast<int>() != 1)
        throw std::invalid_argument("Unsupported checkpoint graph version");
    const PreserveGraphBuilder preserve_builder;
    Graph graph;
    graph.order = state["order"].cast<std::vector<std::string>>();
    graph.timestep = state["timestep"].cast<int>();
    const auto start = state["start_idx"].cast<std::vector<int>>();
    if (start.size() != GraphNode::status_num + 1)
        throw std::invalid_argument("Invalid checkpoint graph status offsets");
    graph.graph_nodes.clear();
    for (const auto node : state["nodes"].cast<py::list>())
        graph.graph_nodes.push_back(LoadNode(py::cast<py::tuple>(node)));
    const int size = graph.graph_nodes.size();
    if (size < GraphNode::start || !std::is_sorted(start.begin(), start.end()) ||
        start.front() < 0 || start.back() > size)
        throw std::invalid_argument("Invalid checkpoint graph layout");
    for (const auto& node : graph.graph_nodes) {
        if (node.idx < 0 || node.idx >= size)
            throw std::invalid_argument("Invalid checkpoint node index");
        for (int dependency : node.dependency)
            if (dependency < 0 || dependency >= size)
                throw std::invalid_argument("Invalid checkpoint node dependency");
    }
    std::copy(start.begin(), start.end(), graph.start_idx);
    graph.utility = LoadNode(state["utility"].cast<py::tuple>());
    graph.opponent_reach_prob = LoadNode(state["opponent_reach_prob"].cast<py::tuple>());
    graph.reach_prob = LoadNode(state["reach_prob"].cast<py::tuple>());
    graph.action_set_size = LoadNode(state["action_set_size"].cast<py::tuple>());
    graph.subtree_size = LoadNode(state["subtree_size"].cast<py::tuple>());
    graph.BindOwnership();
    return graph;
}

py::dict Checkpoint::SaveRandomState() {
    py::dict state;
    state["version"] = 1;
    state["generator"] = SaveRandomComponent(Basic::generator);
    state["uniform"] = SaveRandomComponent(Basic::uniform);
    state["normal"] = SaveRandomComponent(Basic::normal);
    state["exponential"] = SaveRandomComponent(Basic::exponential);
    return state;
}

void Checkpoint::LoadRandomState(const py::dict& state) {
    if (state["version"].cast<int>() != 1)
        throw std::invalid_argument("Unsupported checkpoint random-state version");
    // Parse everything before mutating the live RNG, so invalid state is atomic.
    const auto generator = LoadRandomComponent<std::default_random_engine>(state["generator"]);
    const auto uniform = LoadRandomComponent<std::uniform_real_distribution<double>>(state["uniform"]);
    const auto normal = LoadRandomComponent<std::normal_distribution<double>>(state["normal"]);
    const auto exponential = LoadRandomComponent<std::exponential_distribution<double>>(state["exponential"]);
    Basic::generator = generator;
    Basic::uniform = uniform;
    Basic::normal = normal;
    Basic::exponential = exponential;
}
