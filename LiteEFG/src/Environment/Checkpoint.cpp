#include "Environment/Checkpoint.h"
#include "Computation/Checkpoint.h"

#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::vector<double> Values(const Vector& vector) {
    std::vector<double> result(vector.size);
    for (int i = 0; i < vector.size; ++i) result[i] = vector[i];
    return result;
}

Vector ReadVector(py::handle value) {
    return Vector(py::cast<std::vector<double>>(value));
}

struct PreserveBuilder {
    GraphBuilderContext context = GraphNode::CaptureContext();
    ~PreserveBuilder() {
        GraphNode::RestoreContext(context);
    }
};

class SavedNode : public Node {
public:
    std::vector<double> utilities;
    explicit SavedNode(int players) : Node(0, 0, players) {}
    double GetUtility(const int& player) override { return utilities.at(player); }
};

class SavedEnvironment : public Environment {
public:
    // The virtual root is owned by Environment; all other nodes live here.
    std::vector<std::unique_ptr<SavedNode>> storage;
    explicit SavedEnvironment(int players) : Environment(players) {}
};

py::object OperationDefinition(py::object state) {
    if (state.is_none()) return state;
    py::dict definition = state.cast<py::dict>();
    definition.attr("pop")("tmp", py::none());
    if (definition.contains("aggregator"))
        definition["aggregator"] = OperationDefinition(definition["aggregator"]);
    return definition;
}

bool SameGraph(const Graph& left, const Graph& right) {
    if (left.graph_nodes.size() != right.graph_nodes.size() || left.order != right.order)
        return false;
    for (size_t i = 0; i < left.graph_nodes.size(); ++i) {
        const auto& a = left.graph_nodes[i];
        const auto& b = right.graph_nodes[i];
        if (a.idx != b.idx || a.order != b.order || a.status != b.status ||
            a.color != b.color || a.dependency != b.dependency)
            return false;
        auto a_definition = OperationDefinition(Checkpoint::SaveNode(a)[5]);
        auto b_definition = OperationDefinition(Checkpoint::SaveNode(b)[5]);
        if (!a_definition.equal(b_definition)) return false;
    }
    return true;
}

void CheckBoundGraph(const Environment& env, const Graph& graph) {
    if (!env.Flags_Initialized || env.num_colors == 0)
        throw std::invalid_argument("Call env.set_graph(graph) before saving a checkpoint");
    Graph mapped = graph;
    std::map<int, int> colors;
    mapped.UpdateColorMapping(colors);
    if (colors != env.color_mapping || !SameGraph(mapped, env.graph))
        throw std::invalid_argument("Checkpoint graph does not match the graph bound to this environment");
}

py::dict SaveGameNode(Node& node, int players) {
    py::dict state;
    state["player"] = node.player;
    state["infoset"] = node.infoset;
    state["idx"] = node.idx;
    state["next_node"] = node.next_node;
    state["parent"] = node.parent;
    state["parent_infoset"] = node.parent_infoset;
    state["chance"] = Values(node.chance);
    state["is_terminal"] = node.is_terminal;
    std::vector<double> utilities(players + 1, 0.0);
    if (node.is_terminal)
        for (int player = 1; player <= players; ++player)
            utilities[player] = node.GetUtility(player);
    state["utilities"] = utilities;
    return state;
}

py::dict SaveInfoset(const Infoset& infoset) {
    py::dict state;
    state["graph"] = Checkpoint::SaveGraph(infoset.graph);
    state["children"] = infoset.children;
    state["parent"] = infoset.parent;
    state["parent_sequences"] = infoset.parent_sequences;
    state["first_visited"] = infoset.first_visited;
    state["player"] = infoset.player;
    state["size"] = infoset.size;
    state["reach"] = infoset.reach;
    state["aggregator_dependency"] = infoset.aggregator_dependency;
    state["is_aggregator"] = infoset.is_aggregator;
    py::list results;
    for (const auto& slot : infoset.results) {
        py::list vectors;
        for (const auto& vector : slot) vectors.append(Values(vector));
        results.append(vectors);
    }
    state["results"] = results;
    return state;
}

void ReadInfoset(Infoset& infoset, const py::dict& state) {
    infoset.graph = Checkpoint::LoadGraph(state["graph"].cast<py::dict>());
    infoset.children = state["children"].cast<std::vector<std::vector<int>>>();
    infoset.parent = state["parent"].cast<std::pair<int, int>>();
    infoset.parent_sequences = state["parent_sequences"].cast<std::vector<std::vector<std::pair<int, int>>>>();
    infoset.first_visited = state["first_visited"].cast<int>();
    infoset.player = state["player"].cast<int>();
    infoset.size = state["size"].cast<int>();
    infoset.reach = state["reach"].cast<double>();
    infoset.aggregator_dependency = state["aggregator_dependency"].cast<std::vector<int>>();
    infoset.is_aggregator = state["is_aggregator"].cast<std::vector<bool>>();
    for (auto slot : state["results"].cast<py::list>()) {
        std::vector<Vector> vectors;
        for (auto value : py::cast<py::list>(slot)) vectors.push_back(ReadVector(value));
        infoset.results.push_back(std::move(vectors));
    }
    if (!infoset.results.empty() &&
        (infoset.results.size() != infoset.graph.graph_nodes.size() ||
         infoset.aggregator_dependency.size() != infoset.results.size() ||
         infoset.is_aggregator.size() != infoset.results.size()))
        throw std::invalid_argument("Invalid checkpoint infoset dimensions");
}

py::dict SaveSequence(const SequenceForm& sequence) {
    py::dict state;
    state["start_sequence"] = sequence.start_sequence;
    state["end_sequence"] = sequence.end_sequence;
    state["strategy"] = Values(sequence.strategy);
    state["gradient"] = Values(sequence.gradient);
    state["counterfactual_value"] = Values(sequence.counterfactual_value);
    state["strategy_idx_map"] = sequence.strategy_idx_map;
    py::list histories;
    for (const auto& history : sequence.history_version_strategies) {
        histories.append(py::make_tuple(Values(history.last_iterate),
            Values(history.best_iterate), Values(history.avg_iterate),
            Values(history.linear_avg_iterate), history.best_exploitability, history.timestep));
    }
    state["histories"] = histories;
    return state;
}

void ReadSequence(SequenceForm& sequence, const py::dict& state) {
    if (sequence.start_sequence != state["start_sequence"].cast<std::vector<int>>() ||
        sequence.end_sequence != state["end_sequence"].cast<std::vector<int>>())
        throw std::invalid_argument("Invalid checkpoint sequence layout");
    const int size = sequence.strategy.size;
    sequence.strategy = ReadVector(state["strategy"]);
    sequence.gradient = ReadVector(state["gradient"]);
    sequence.counterfactual_value = ReadVector(state["counterfactual_value"]);
    sequence.strategy_idx_map = state["strategy_idx_map"].cast<std::vector<int>>();
    if (sequence.strategy.size != size || sequence.gradient.size != size ||
        sequence.counterfactual_value.size != size)
        throw std::invalid_argument("Invalid checkpoint sequence dimensions");
    for (auto item : state["histories"].cast<py::list>()) {
        auto values = py::cast<py::tuple>(item);
        if (values.size() != 6) throw std::invalid_argument("Invalid checkpoint strategy history");
        HistoryVersionStrategy history(size);
        history.last_iterate = ReadVector(values[0]);
        history.best_iterate = ReadVector(values[1]);
        history.avg_iterate = ReadVector(values[2]);
        history.linear_avg_iterate = ReadVector(values[3]);
        history.best_exploitability = values[4].cast<double>();
        history.timestep = values[5].cast<double>();
        if (history.last_iterate.size != size || history.best_iterate.size != size ||
            history.avg_iterate.size != size || history.linear_avg_iterate.size != size)
            throw std::invalid_argument("Invalid checkpoint history dimensions");
        sequence.history_version_strategies.push_back(std::move(history));
    }
    for (int idx : sequence.strategy_idx_map)
        if (idx < -1 || idx >= int(sequence.history_version_strategies.size()))
            throw std::invalid_argument("Invalid checkpoint strategy history index");
}

std::vector<std::pair<int, int>> TraversedInfosets(const Environment& env) {
    std::vector<std::pair<int, int>> result;
    for (const auto* infoset : env.traverse_infoset) {
        bool found = false;
        for (int player = 1; player <= env.player_num && !found; ++player)
            for (size_t i = 0; i < env.infosets[player].size(); ++i)
                if (infoset == &env.infosets[player][i]) {
                    result.emplace_back(player, int(i));
                    found = true;
                    break;
                }
        if (!found) throw std::invalid_argument("Invalid environment traversal infoset");
    }
    return result;
}

}

py::dict Checkpoint::SaveEnvironment(const Environment& env, const Graph& graph) {
    CheckBoundGraph(env, graph);
    py::dict state;
    state["version"] = 1;
    state["player_num"] = env.player_num;
    state["traverse"] = env.traverse;
    state["aggregate_opponents"] = env.Is_Aggregate_Opponents;
    state["graph"] = SaveGraph(env.graph);
    state["color_mapping"] = env.color_mapping;
    state["num_colors"] = env.num_colors;
    state["is_color_to_update"] = env.is_color_to_update;
    state["infoset_names"] = env.infoset_names;
    py::list nodes, infosets, sequences;
    for (auto* node : env.nodes) {
        auto saved = SaveGameNode(*node, env.player_num);
        saved["reach"] = Values(node->reach);
        nodes.append(saved);
    }
    for (const auto& player_infosets : env.infosets) {
        py::list group;
        for (const auto& infoset : player_infosets) group.append(SaveInfoset(infoset));
        infosets.append(group);
    }
    for (const auto& sequence : env.sequence_form_strategies)
        sequences.append(SaveSequence(sequence));
    state["nodes"] = nodes;
    state["infosets"] = infosets;
    state["sequences"] = sequences;
    std::vector<int> traversal;
    for (const auto* node : env.traverse_order) traversal.push_back(node->idx);
    state["traverse_order"] = traversal;
    state["traverse_infoset"] = TraversedInfosets(env);
    return state;
}

std::shared_ptr<Environment> Checkpoint::LoadEnvironment(const py::dict& state) {
    const PreserveBuilder preserve_builder;
    if (state["version"].cast<int>() != 1)
        throw std::invalid_argument("Unsupported checkpoint environment version");
    const int players = state["player_num"].cast<int>();
    if (players < 1) throw std::invalid_argument("Invalid checkpoint player count");
    auto env = std::make_shared<SavedEnvironment>(players);
    env->traverse = state["traverse"].cast<int>();
    if (env->traverse < Environment::Enumerate || env->traverse > Environment::External)
        throw std::invalid_argument("Invalid checkpoint traversal");
    env->graph = LoadGraph(state["graph"].cast<py::dict>());
    env->Is_Aggregate_Opponents = state["aggregate_opponents"].cast<bool>();
    env->color_mapping = state["color_mapping"].cast<std::map<int, int>>();
    env->num_colors = state["num_colors"].cast<int>();
    env->is_color_to_update = state["is_color_to_update"].cast<std::vector<bool>>();
    env->infoset_names = state["infoset_names"].cast<std::vector<std::vector<std::string>>>();
    if (env->num_colors < 1 || env->is_color_to_update.size() != size_t(env->num_colors) ||
        env->infoset_names.size() != size_t(players + 1))
        throw std::invalid_argument("Invalid checkpoint environment dimensions");

    auto nodes = state["nodes"].cast<py::list>();
    if (nodes.empty()) throw std::invalid_argument("Checkpoint has no game tree");
    // Mark ownership as soon as the virtual root exists, so exceptions clean up.
    env->nodes.push_back(new Node(0, 0, players));
    env->Flags_Initialized = true;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto saved = nodes[i].cast<py::dict>();
        if (i > 0) {
            env->storage.push_back(std::make_unique<SavedNode>(players));
            env->nodes.push_back(env->storage.back().get());
            env->storage.back()->utilities = saved["utilities"].cast<std::vector<double>>();
            if (env->storage.back()->utilities.size() != size_t(players + 1))
                throw std::invalid_argument("Invalid checkpoint utilities");
        }
        auto& node = *env->nodes[i];
        node.player = saved["player"].cast<int>();
        node.infoset = saved["infoset"].cast<int>();
        node.idx = saved["idx"].cast<int>();
        node.next_node = saved["next_node"].cast<std::vector<int>>();
        node.parent = saved["parent"].cast<std::pair<int, int>>();
        node.parent_infoset = saved["parent_infoset"].cast<std::vector<std::pair<int, int>>>();
        node.chance = ReadVector(saved["chance"]);
        node.reach = ReadVector(saved["reach"]);
        node.is_terminal = saved["is_terminal"].cast<bool>();
        if (node.idx != int(i) || node.player < 0 || node.player > players ||
            node.parent_infoset.size() != size_t(players + 1))
            throw std::invalid_argument("Invalid checkpoint game node");
        for (int child : node.next_node)
            if (child <= int(i) || child >= int(nodes.size()))
                throw std::invalid_argument("Invalid checkpoint game edge");
    }
    auto infosets = state["infosets"].cast<py::list>();
    auto sequences = state["sequences"].cast<py::list>();
    if (infosets.size() != size_t(players + 1) || sequences.size() != infosets.size())
        throw std::invalid_argument("Invalid checkpoint player dimensions");
    env->infosets.resize(players + 1);
    for (int player = 0; player <= players; ++player) {
        auto group = infosets[player].cast<py::list>();
        if (group.empty()) throw std::invalid_argument("Missing checkpoint root infoset");
        env->infosets[player].resize(group.size());
        for (size_t i = 0; i < group.size(); ++i)
            ReadInfoset(env->infosets[player][i], group[i].cast<py::dict>());
        env->sequence_form_strategies.emplace_back(&env->infosets[player]);
        ReadSequence(env->sequence_form_strategies.back(), sequences[player].cast<py::dict>());
    }
    for (int idx : state["traverse_order"].cast<std::vector<int>>())
        env->traverse_order.push_back(env->nodes.at(idx));
    for (auto index : state["traverse_infoset"].cast<std::vector<std::pair<int, int>>>())
        env->traverse_infoset.push_back(&env->infosets.at(index.first).at(index.second));
    return env;
}

void Checkpoint::Restore(Environment& env, Graph& graph,
                         const Environment& saved_env, const Graph& saved_graph) {
    CheckBoundGraph(env, graph);
    CheckBoundGraph(saved_env, saved_graph);
    if (!SameGraph(graph, saved_graph))
        throw std::invalid_argument("Checkpoint graph structure does not match this graph");
    if (env.player_num != saved_env.player_num || env.nodes.size() != saved_env.nodes.size() ||
        env.infoset_names != saved_env.infoset_names)
        throw std::invalid_argument("Checkpoint game does not match this environment");
    for (size_t i = 0; i < env.nodes.size(); ++i)
        if (!SaveGameNode(*env.nodes[i], env.player_num).equal(
                SaveGameNode(*saved_env.nodes[i], saved_env.player_num)))
            throw std::invalid_argument("Checkpoint game does not match this environment");

    // All compatibility checks precede mutation. Copies own their operations
    // and values; transient pointers are rebound to the destination storage.
    auto infosets = saved_env.infosets;
    auto sequences = saved_env.sequence_form_strategies;
    const auto traversed = TraversedInfosets(saved_env);
    env.infosets = std::move(infosets);
    env.ResetParallelSchedule();
    env.sequence_form_strategies = std::move(sequences);
    for (int player = 0; player <= env.player_num; ++player)
        env.sequence_form_strategies[player].infosets = &env.infosets[player];
    env.graph = saved_env.graph;
    env.traverse = saved_env.traverse;
    env.Is_Aggregate_Opponents = saved_env.Is_Aggregate_Opponents;
    env.color_mapping = saved_env.color_mapping;
    env.num_colors = saved_env.num_colors;
    env.is_color_to_update = saved_env.is_color_to_update;
    for (size_t i = 0; i < env.nodes.size(); ++i) env.nodes[i]->reach = saved_env.nodes[i]->reach;
    env.traverse_order.clear();
    for (auto* node : saved_env.traverse_order) env.traverse_order.push_back(env.nodes[node->idx]);
    env.traverse_infoset.clear();
    for (auto index : traversed) env.traverse_infoset.push_back(&env.infosets[index.first][index.second]);
    graph = saved_graph;
}
