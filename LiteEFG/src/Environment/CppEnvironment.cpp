#include "Computation/Graph.h"
#include "Basic/BasicFunction.h"
#include "LiteEFG/Game.h"

#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace py = pybind11;
namespace {
using Results = std::vector<std::vector<Vector>>;
using Sequence = std::pair<int, int>;

// Only information sets survive a traversal. Graph programs and numeric layouts
// are shared; an information set holds one packed numeric row, not a Graph.
class CppEnvironment {
    struct Layout { std::vector<size_t> offsets; };
    struct Info {
        std::string key;
        int player, parent, parent_action, depth, layout = -1;
        size_t sequence;
        std::vector<int> actions;
        std::vector<double> values, average, linear_average, last;
    };
    struct Visit {
        Results results;
        std::vector<std::vector<int>> children;
    };
    void* library = nullptr;
    liteefg::Game* game = nullptr;
    void (*destroy)(liteefg::Game*) = nullptr;
    int players = 0;
    std::string traversal;
    std::unique_ptr<Graph> graph;
    std::vector<Info> infos;
    std::vector<std::unordered_map<std::string, int>> lookup;
    std::vector<Layout> layouts;
    std::map<std::vector<int>, int> layout_ids;
    std::map<int, Results> initial_rows;
    std::map<int, size_t> static_parent_sources;
    std::unordered_map<int, std::vector<Vector>> initial_parent_values;
    std::map<int, int> colors;
    std::vector<bool> enabled;
    std::unordered_map<int, Visit> visits;
    std::vector<int> touched;
    std::vector<int> average_nodes;
    uint64_t snapshots = 0, update_nodes = 0, evaluation_nodes = 0;
    size_t sequences = 0, peak_depth = 0;
    int result_count = 0;
    bool cache_initial_rows = true;

    static void Require(bool ok, const std::string& message) {
        if (!ok) throw std::invalid_argument(message);
    }
    static void CheckMode(const std::string& mode) {
        Require(mode == "Enumerate" || mode == "External" || mode == "Outcome",
                "traverse_type must be Enumerate, External, or Outcome");
    }
    void Ready() const { Require(bool(graph), "Call set_graph before using CppEnv"); }
    std::vector<int> Nodes(const std::vector<GraphNode>& nodes) const {
        Ready();
        Require(nodes.size() == size_t(players), "One strategy node is required per player");
        std::vector<int> out;
        for (auto& node : nodes) {
            Require(node.idx >= 0 && node.idx < result_count, "Invalid graph node");
            out.push_back(node.idx);
        }
        return out;
    }
    Results Empty(int n) const {
        Results row(result_count, std::vector<Vector>(1));
        row[0][0] = Vector(n, 0.0);
        row[1][0] = Vector(1, double(n));
        row[2][0] = Vector(1, 0.0);
        row[3][0] = Vector(1, 0.0);
        row[4][0] = Vector(n, 0.0);
        return row;
    }
    Vector Read(int id, int node) const {
        const auto& e = infos.at(id);
        const auto& off = layouts[e.layout].offsets;
        Vector out(int(off[node + 1] - off[node]), 0.0);
        for (int k = 0; k < out.size; ++k) out[k] = e.values[off[node] + k];
        return out;
    }
    Results Load(int id) const {
        Results row(result_count, std::vector<Vector>(1));
        for (int j = 0; j < result_count; ++j) row[j][0] = Read(id, j);
        return row;
    }
    void Save(int id, const Results& row) {
        std::vector<int> sizes;
        for (auto& v : row) sizes.push_back(v[0].size);
        auto found = layout_ids.find(sizes);
        int layout;
        if (found == layout_ids.end()) {
            layout = int(layouts.size());
            Layout item;
            item.offsets.push_back(0);
            for (int n : sizes) item.offsets.push_back(item.offsets.back() + n);
            layouts.push_back(std::move(item));
            layout_ids.emplace(std::move(sizes), layout);
        } else layout = found->second;
        auto& e = infos[id];
        e.layout = layout;
        e.values.resize(layouts[layout].offsets.back());
        size_t pos = 0;
        for (auto& v : row) for (int k = 0; k < v[0].size; ++k) e.values[pos++] = v[0][k];
    }
    Vector Value(int id, int node) const {
        auto active = visits.find(id);
        return active == visits.end() ? Read(id, node) : active->second.results[node][0];
    }
    void Execute(int id, Results& row, int status, bool initializing = false) {
        const auto& entry = infos[id];
        for (const auto& op : graph->graph_nodes) {
            if (!op.operation || op.status != status || (!initializing && !enabled[op.color])) continue;
            if (op.operation->name == "Aggregate") {
                auto* agg = dynamic_cast<AggregateOperation*>(op.operation.get());
                int source = op.dependency.at(0);
                const double padding = agg->info[AggregateOperation::padding];
                Vector output;
                if (agg->info[AggregateOperation::object] < 0) {
                    output = Vector(1, padding);
                    if (entry.parent >= 0) {
                        auto parent = initializing
                            ? initial_parent_values.at(entry.parent).at(static_parent_sources.at(source))
                            : Value(entry.parent, source);
                        Require(parent.size == 1 || parent.size == int(infos[entry.parent].actions.size()),
                                "Parent aggregate requires a scalar or per-action vector");
                        output[0] = parent[parent.size == 1 ? 0 : entry.parent_action];
                    }
                } else {
                    output = Vector(int(entry.actions.size()), padding);
                    auto active = visits.find(id);
                    if (active != visits.end()) {
                        for (size_t a = 0; a < entry.actions.size(); ++a) {
                            Vector input;
                            for (int child : active->second.children[a]) input.Concat(Value(child, source));
                            if (input.size) {
                                Vector reduced;
                                agg->aggregator->Execute(reduced, {&input});
                                output[a] = reduced[0];
                            }
                        }
                    }
                }
                row[op.idx][0] = output;
            } else {
                std::vector<Vector*> input;
                for (int dep : op.dependency) input.push_back(&row[dep][0]);
                op.operation->Execute(row[op.idx][0], input);
            }
        }
    }
    std::vector<double> Policy(int id, int node, const std::string& type = "default") const {
        const auto& e = infos[id];
        std::vector<double> out;
        if (type == "default") {
            auto v = Read(id, node);
            for (int a = 0; a < v.size; ++a) out.push_back(v[a]);
        } else {
            Require(snapshots && average_nodes[e.player - 1] == node,
                    "Call update_strategy with this strategy before selecting its history");
            if (type == "last-iterate") out = e.last;
            else if (type == "avg-iterate" || type == "linear-avg-iterate") {
                out = type == "avg-iterate" ? e.average : e.linear_average;
                double sum = std::accumulate(out.begin(), out.end(), 0.0);
                if (sum > 0) for (auto& x : out) x /= sum;
                else {
                    // An unreachable sequence may use any legal behavior.
                    out.assign(e.actions.size(), 1.0 / e.actions.size());
                }
            } else throw std::invalid_argument("CppEnv strategy selector must be default, last-iterate, avg-iterate, or linear-avg-iterate");
        }
        Require(out.size() == e.actions.size(), "Strategy dimension does not match legal actions");
        double sum = 0;
        for (auto x : out) {
            Require(std::isfinite(x) && x >= 0, "Strategy probabilities must be finite and nonnegative");
            sum += x;
        }
        Require(std::abs(sum - 1) < 1e-8, "Strategy probabilities must sum to one");
        return out;
    }
    void InitializeAverage(int id) {
        auto& e = infos[id];
        auto policy = Policy(id, average_nodes[e.player - 1]);
        e.last = policy;
        double mass = double(snapshots), linear = double(snapshots) * (double(snapshots) + 1) / 2;
        if (e.parent >= 0) {
            mass = infos[e.parent].average[e.parent_action];
            linear = infos[e.parent].linear_average[e.parent_action];
        }
        e.average = policy;
        e.linear_average = policy;
        for (size_t a = 0; a < policy.size(); ++a) {
            e.average[a] *= mass;
            e.linear_average[a] *= linear;
        }
    }
    int Register(liteefg::State& state, int player, const std::vector<Sequence>& last) {
        Require(player >= 1 && player <= players, "Invalid CurrentPlayer from C++ game");
        auto key = state.InformationSet();
        auto actions = state.LegalActions();
        Require(!actions.empty(), "A decision state must have legal actions");
        auto found = lookup[player].find(key);
        if (found != lookup[player].end()) {
            const auto& e = infos[found->second];
            Require(e.actions == actions, "Information set has inconsistent ordered legal actions");
            Require(e.parent == last[player].first && e.parent_action == last[player].second,
                    "Information set violates perfect recall: inconsistent own parent sequence");
            return found->second;
        }
        Require(std::unordered_set<int>(actions.begin(), actions.end()).size() == actions.size(),
                "Duplicate legal action IDs");
        int id = int(infos.size());
        Info entry;
        entry.key = key;
        entry.player = player;
        entry.parent = last[player].first;
        entry.parent_action = last[player].second;
        entry.depth = entry.parent < 0 ? 0 : infos[entry.parent].depth + 1;
        entry.sequence = sequences;
        sequences += actions.size();
        entry.actions = std::move(actions);
        infos.push_back(std::move(entry));
        lookup[player].emplace(std::move(key), id);
        int n = int(infos[id].actions.size());
        Results row;
        auto cached = initial_rows.find(n);
        if (cache_initial_rows && cached != initial_rows.end()) row = cached->second;
        else {
            row = Empty(n);
            Execute(id, row, GraphNode::static_backward_node, true);
            Execute(id, row, GraphNode::static_forward_node, true);
            if (cache_initial_rows) initial_rows.emplace(n, row);
        }
        Save(id, row);
        if (!static_parent_sources.empty()) {
            std::vector<Vector> values(static_parent_sources.size());
            for (auto [source, offset] : static_parent_sources) values[offset] = row[source][0];
            initial_parent_values.emplace(id, std::move(values));
        }
        if (!average_nodes.empty()) InitializeAverage(id);
        return id;
    }
    Visit& Touch(int id, double own, double opponents) {
        auto it = visits.find(id);
        if (it == visits.end()) {
            Visit v;
            v.results = Load(id);
            v.results[0][0].Set(0);
            v.results[2][0] = Vector(1, own);
            v.results[3][0] = Vector(1, 0.0);
            v.children.resize(infos[id].actions.size());
            it = visits.emplace(id, std::move(v)).first;
            touched.push_back(id);
        }
        it->second.results[3][0][0] += opponents;
        return it->second;
    }
    static std::vector<std::pair<int, double>> Chance(liteefg::State& state) {
        auto outcomes = state.ChanceOutcomes();
        Require(!outcomes.empty(), "Empty chance distribution");
        double sum = 0;
        std::unordered_set<int> ids;
        for (auto [a, p] : outcomes) {
            Require(ids.insert(a).second && std::isfinite(p) && p >= 0, "Invalid chance distribution");
            sum += p;
        }
        Require(std::abs(sum - 1) < 1e-8, "Chance probabilities must sum to one");
        return outcomes;
    }
    std::vector<double> Returns(liteefg::State& state) const {
        auto values = state.Returns();
        Require(values.size() == size_t(players), "Returns must contain one payoff per player");
        for (double value : values) Require(std::isfinite(value), "Payoffs must be finite");
        return values;
    }
    void Walk(liteefg::State& state, const std::vector<int>& nodes, int target,
              const std::string& mode, std::vector<Sequence>& last,
              std::vector<double>& reach, size_t depth) {
        ++update_nodes;
        if ((update_nodes & 0xfffff) == 0 && PyErr_CheckSignals() != 0) throw py::error_already_set();
        peak_depth = std::max(peak_depth, depth);
        int player = state.CurrentPlayer();
        if (player == -1) {
            auto returns = Returns(state);
            for (int p = 1; p <= players; ++p) if ((target == -1 || p == target) && last[p].first >= 0) {
                double weight = 1;
                if (mode == "Enumerate") for (int j = 0; j <= players; ++j) if (j != p) weight *= reach[j];
                visits.at(last[p].first).results[0][0][last[p].second] += returns[p - 1] * weight;
            }
            return;
        }
        if (player == 0) {
            std::vector<std::pair<int, double>> branches;
            if (mode == "Enumerate") branches = Chance(state);
            else {
                int a = state.SampleChance(Basic::uniform(Basic::generator));
                double p = state.ChanceProbability(a);
                Require(std::isfinite(p) && p > 0 && p <= 1, "Invalid sampled chance probability");
                branches.emplace_back(a, p);
            }
            for (auto [a, prob] : branches) {
                if (prob == 0) continue;
                double saved = reach[0];
                reach[0] *= prob;
                state.ApplyAction(a);
                Walk(state, nodes, target, mode, last, reach, depth + 1);
                state.UndoAction();
                reach[0] = saved;
            }
            return;
        }
        int id = Register(state, player, last);
        auto policy = Policy(id, nodes[player - 1]);
        if (target == -1 || player == target) {
            double opponents = 1;
            for (int j = 0; j <= players; ++j) if (j != player) opponents *= reach[j];
            Touch(id, reach[player], opponents);
        }
        const auto previous = last[player];
        double previous_reach = reach[player];
        auto actions = infos[id].actions;
        int sampled = -1;
        if (mode == "Outcome" || (mode == "External" && player != target)) sampled = Basic::Sample(policy);
        for (int a = 0; a < int(actions.size()); ++a) {
            if (sampled >= 0 && sampled != a) continue;
            last[player] = {id, a};
            reach[player] = previous_reach * policy[a];
            state.ApplyAction(actions[a]);
            Walk(state, nodes, target, mode, last, reach, depth + 1);
            state.UndoAction();
        }
        last[player] = previous;
        reach[player] = previous_reach;
    }
    void RunVisits() {
        std::stable_sort(touched.begin(), touched.end(), [this](int a, int b) {
            return infos[a].depth < infos[b].depth;
        });
        for (int id : touched) {
            const auto& e = infos[id];
            auto parent = visits.find(e.parent);
            if (parent != visits.end()) parent->second.children[e.parent_action].push_back(id);
        }
        for (auto it = touched.rbegin(); it != touched.rend(); ++it)
            Execute(*it, visits.at(*it).results, GraphNode::backward_node);
        // Forward child aggregation uses the preceding forward values, matching
        // the explicit backend's gather-before-forward-update semantics.
        for (int id : touched) {
            // Child sources have not yet been updated when a parent is visited
            // in depth order. Parent aggregates see this pass's parent values.
            Execute(id, visits.at(id).results, GraphNode::forward_node);
        }
        for (int id : touched) Save(id, visits.at(id).results);
        visits.clear();
        touched.clear();
    }
    void EvaluateWalk(liteefg::State& state, int target, const std::vector<int>& nodes,
                      const std::string& type, std::vector<Sequence>& last, double weight,
                      std::vector<double>& gradient, double& root, size_t depth) {
        ++evaluation_nodes;
        if ((evaluation_nodes & 0xfffff) == 0 && PyErr_CheckSignals() != 0) throw py::error_already_set();
        peak_depth = std::max(peak_depth, depth);
        int player = state.CurrentPlayer();
        if (player == -1) {
            double value = Returns(state)[target - 1] * weight;
            const auto seq = last[target];
            if (seq.first < 0) root += value;
            else {
                if (gradient.size() < sequences) gradient.resize(sequences, 0);
                gradient[infos[seq.first].sequence + seq.second] += value;
            }
            return;
        }
        if (player == 0) {
            for (auto [action, prob] : Chance(state)) if (prob > 0) {
                state.ApplyAction(action);
                EvaluateWalk(state, target, nodes, type, last, weight * prob, gradient, root, depth + 1);
                state.UndoAction();
            }
            return;
        }
        int id = Register(state, player, last);
        auto actions = infos[id].actions;
        std::vector<double> policy;
        if (player != target) policy = Policy(id, nodes[player - 1], type);
        const auto previous = last[player];
        for (size_t a = 0; a < actions.size(); ++a) {
            double p = player == target ? 1 : policy[a];
            if (p == 0) continue;
            last[player] = {id, int(a)};
            state.ApplyAction(actions[a]);
            EvaluateWalk(state, target, nodes, type, last, weight * p, gradient, root, depth + 1);
            state.UndoAction();
        }
        last[player] = previous;
    }
    std::pair<std::vector<double>, std::vector<double>> Evaluate(const std::vector<GraphNode>& strategies,
                                                               const std::string& type) {
        auto nodes = Nodes(strategies);
        Require(type == "default" || type == "last-iterate" || type == "avg-iterate" || type == "linear-avg-iterate",
                "Unsupported CppEnv strategy selector");
        if (type != "default") Require(snapshots && nodes == average_nodes, "No recorded history for this strategy");
        evaluation_nodes = 0;
        std::vector<double> utilities(players), gains(players);
        for (int p = 1; p <= players; ++p) {
            std::vector<double> gradient(sequences, 0);
            std::vector<Sequence> last(players + 1, {-1, -1});
            double root = 0;
            auto state = game->NewInitialState();
            EvaluateWalk(*state, p, nodes, type, last, 1, gradient, root, 0);
            gradient.resize(sequences, 0);
            // Parent entries are registered before descendants even when some
            // information sets are first encountered during this evaluation.
            std::vector<double> realization(sequences, 0);
            double utility = root;
            for (int id = 0; id < int(infos.size()); ++id) if (infos[id].player == p) {
                const auto& e = infos[id];
                double reach = e.parent < 0 ? 1 : realization[infos[e.parent].sequence + e.parent_action];
                auto policy = Policy(id, nodes[p - 1], type);
                for (size_t a = 0; a < e.actions.size(); ++a) {
                    realization[e.sequence + a] = reach * policy[a];
                    utility += realization[e.sequence + a] * gradient[e.sequence + a];
                }
            }
            double best = root;
            for (int id = int(infos.size()) - 1; id >= 0; --id) if (infos[id].player == p) {
                const auto& e = infos[id];
                double value = *std::max_element(gradient.begin() + e.sequence,
                                                gradient.begin() + e.sequence + e.actions.size());
                if (e.parent < 0) best += value;
                else gradient[infos[e.parent].sequence + e.parent_action] += value;
            }
            utilities[p - 1] = utility;
            gains[p - 1] = best - utility;
        }
        return {utilities, gains};
    }
public:
    CppEnvironment(const std::string& path, const std::string& parameters, const std::string& mode) : traversal(mode) {
        CheckMode(mode);
#ifdef _WIN32
        throw std::runtime_error("CppEnv currently requires Linux/WSL or macOS");
#else
        library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(std::string("Cannot load C++ game: ") + dlerror());
        try {
            auto version = reinterpret_cast<int (*)()>(dlsym(library, "liteefg_game_api_version"));
            auto create = reinterpret_cast<liteefg::Game* (*)(const char*)>(dlsym(library, "liteefg_create_game"));
            destroy = reinterpret_cast<void (*)(liteefg::Game*)>(dlsym(library, "liteefg_destroy_game"));
            Require(version && create && destroy, "C++ game is missing LITEEFG_REGISTER_GAME exports");
            Require(version() == liteefg::kGameApiVersion, "Incompatible C++ game API version");
            game = create(parameters.c_str());
            Require(game, "C++ game factory returned null");
            players = game->NumPlayers();
            Require(players > 0, "A game must have at least one player");
            lookup.resize(players + 1);
        } catch (...) {
            if (game) destroy(game);
            dlclose(library);
            throw;
        }
#endif
    }
    ~CppEnvironment() {
        if (game) destroy(game);
#ifndef _WIN32
        if (library) dlclose(library);
#endif
    }
    void SetGraph(const Graph& program) {
        Require(infos.empty(), "Create a new CppEnv to replace a graph after traversal");
        auto candidate = std::make_unique<Graph>(program);
        cache_initial_rows = true;
        static_parent_sources.clear();
        result_count = 0;
        for (const auto& op : candidate->graph_nodes) {
            result_count = std::max(result_count, op.idx + 1);
            if (op.operation && op.status <= GraphNode::static_forward_node &&
                op.operation->name.find("Random") == 0) cache_initial_rows = false;
            for (int dep : op.dependency) Require(dep != GraphNode::subtree_size,
                "CppEnv cannot infer complete subtree_size; use an algorithm without global subtree weights");
            if (op.operation && op.operation->name == "Aggregate") {
                Require(op.operation->info[AggregateOperation::player] > 0,
                        "CppEnv supports own-player aggregates, not opponent topology aggregates");
                if (op.status <= GraphNode::static_forward_node) {
                    Require(op.operation->info[AggregateOperation::object] < 0,
                            "CppEnv cannot evaluate static children aggregates without complete infoset topology");
                    cache_initial_rows = false;
                    int source = op.dependency.at(0);
                    if (!static_parent_sources.count(source))
                        static_parent_sources.emplace(source, static_parent_sources.size());
                }
            }
        }
        int count = candidate->UpdateColorMapping(colors);
        enabled.assign(count, true);
        candidate->Initialize();
        graph = std::move(candidate);
        initial_rows.clear();
    }
    void Update(const std::vector<GraphNode>& strategies, int player, const std::vector<int>& update_colors,
                const std::string& mode_arg) {
        auto nodes = Nodes(strategies);
        Require(player == -1 || (player >= 1 && player <= players), "Invalid updating player");
        std::string mode = mode_arg == "default" ? traversal : mode_arg;
        CheckMode(mode);
        std::fill(enabled.begin(), enabled.end(), false);
        for (int color : update_colors) {
            if (color == -1) { std::fill(enabled.begin(), enabled.end(), true); break; }
            Require(colors.count(color), "Invalid update color");
            enabled[colors.at(color)] = true;
        }
        update_nodes = 0;
        auto pass = [&](int target) {
            visits.clear(); touched.clear();
            auto state = game->NewInitialState();
            std::vector<Sequence> last(players + 1, {-1, -1});
            std::vector<double> reach(players + 1, 1);
            try { Walk(*state, nodes, target, mode, last, reach, 0); RunVisits(); }
            catch (...) { visits.clear(); touched.clear(); throw; }
        };
        if (mode == "External") {
            for (int p = 1; p <= players; ++p) if (player == -1 || p == player) pass(p);
        } else pass(player);
    }
    void UpdateStrategy(const std::vector<GraphNode>& strategies, bool best) {
        Require(!best, "CppEnv does not maintain best-iterate; call exact exploitability explicitly at evaluation intervals");
        auto nodes = Nodes(strategies);
        Require(average_nodes.empty() || average_nodes == nodes, "CppEnv records one strategy profile; use a separate environment for another profile");
        if (average_nodes.empty()) {
            average_nodes = nodes;
            for (int id = 0; id < int(infos.size()); ++id) InitializeAverage(id);
        }
        std::vector<double> realization(sequences, 0);
        for (int id = 0; id < int(infos.size()); ++id) {
            auto& e = infos[id];
            auto policy = Policy(id, nodes[e.player - 1]);
            double reach = e.parent < 0 ? 1 : realization[infos[e.parent].sequence + e.parent_action];
            for (size_t a = 0; a < e.actions.size(); ++a) {
                double mass = reach * policy[a];
                realization[e.sequence + a] = mass;
                e.average[a] += mass;
                e.linear_average[a] += double(snapshots + 1) * mass;
            }
            e.last = std::move(policy);
        }
        ++snapshots;
    }
    std::vector<double> Exploitability(const std::vector<GraphNode>& nodes, const std::string& type) { return Evaluate(nodes, type).second; }
    std::vector<double> Utility(const std::vector<GraphNode>& nodes, const std::string& type) { return Evaluate(nodes, type).first; }
    py::dict Metrics(const std::vector<GraphNode>& nodes, const std::string& type) {
        auto result = Evaluate(nodes, type);
        py::dict out;
        out["utility"] = result.first; out["deviation_gain"] = result.second;
        out["nash_conv"] = std::accumulate(result.second.begin(), result.second.end(), 0.0);
        return out;
    }
    py::list GetStrategy(int player, const GraphNode& node, const std::string& type) const {
        Ready();
        Require(player >= 1 && player <= players, "Invalid player");
        Require(node.idx >= 0 && node.idx < result_count, "Invalid graph node");
        py::list out;
        for (int id = 0; id < int(infos.size()); ++id) if (infos[id].player == player)
            out.append(py::make_tuple(py::bytes(infos[id].key), Policy(id, node.idx, type)));
        return out;
    }
    py::list GetValue(int player, const GraphNode& node) const {
        Ready(); Require(player >= 1 && player <= players, "Invalid player");
        Require(node.idx >= 0 && node.idx < result_count, "Invalid graph node");
        py::list out;
        for (int id = 0; id < int(infos.size()); ++id) if (infos[id].player == player) {
            auto value = Read(id, node.idx);
            std::vector<double> values;
            for (int a = 0; a < value.size; ++a) values.push_back(value[a]);
            out.append(py::make_tuple(py::bytes(infos[id].key), values));
        }
        return out;
    }
    py::dict Stats() const {
        py::dict out;
        std::vector<size_t> counts;
        for (int p = 1; p <= players; ++p) counts.push_back(lookup[p].size());
        out["game"] = game->Name(); out["players"] = players;
        out["infosets"] = counts; out["information_sets"] = infos.size();
        out["sequences"] = sequences; out["stored_nodes"] = 0;
        out["update_nodes"] = update_nodes; out["evaluation_nodes"] = evaluation_nodes;
        out["peak_depth"] = peak_depth; out["strategy_snapshots"] = snapshots;
        return out;
    }
    py::dict Inspect(const std::vector<int>& actions) const {
        auto state = game->NewInitialState();
        for (int action : actions) {
            auto legal = state->LegalActions();
            Require(std::find(legal.begin(), legal.end(), action) != legal.end(), "Illegal action in inspect");
            state->ApplyAction(action);
        }
        py::dict out;
        int player = state->CurrentPlayer();
        out["current_player"] = player;
        out["legal_actions"] = state->LegalActions();
        out["infoset"] = py::bytes(player > 0 ? state->InformationSet() : "");
        out["returns"] = player == -1 ? Returns(*state) : std::vector<double>(players, 0.0);
        if (player == 0) out["chance_outcomes"] = Chance(*state);
        return out;
    }
    int NumPlayers() const { return players; }
};
} // namespace

void BindCppEnvironment(py::module_& m) {
    auto cls = py::class_<CppEnvironment>(m, "_CppEnv")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("library_path"), py::arg("parameters") = "{}", py::arg("traverse_type") = "External")
        .def("set_graph", &CppEnvironment::SetGraph)
        .def_property_readonly("player_num", &CppEnvironment::NumPlayers)
        .def("stats", &CppEnvironment::Stats)
        .def("inspect", &CppEnvironment::Inspect, py::arg("actions"))
        .def("get_strategy", &CppEnvironment::GetStrategy, py::arg("player"), py::arg("strategy"), py::arg("type_name") = "default")
        .def("get_value", &CppEnvironment::GetValue, py::arg("player"), py::arg("node"));
    cls.def("update", &CppEnvironment::Update, py::arg("strategy"), py::arg("upd_player") = -1,
            py::arg("upd_color") = std::vector<int>{-1}, py::arg("traverse_type") = "default");
    cls.def("update", [](CppEnvironment& env, const GraphNode& node, int player, const std::vector<int>& colors, const std::string& mode) {
        env.Update(std::vector<GraphNode>(env.NumPlayers(), node), player, colors, mode);
    }, py::arg("strategy"), py::arg("upd_player") = -1, py::arg("upd_color") = std::vector<int>{-1}, py::arg("traverse_type") = "default");
    cls.def("update_strategy", &CppEnvironment::UpdateStrategy, py::arg("strategy"), py::arg("update_best") = false);
    cls.def("update_strategy", [](CppEnvironment& env, const GraphNode& node, bool best) {
        env.UpdateStrategy(std::vector<GraphNode>(env.NumPlayers(), node), best);
    }, py::arg("strategy"), py::arg("update_best") = false);
    cls.def("exploitability", &CppEnvironment::Exploitability, py::arg("strategy"), py::arg("type_name") = "default");
    cls.def("exploitability", [](CppEnvironment& env, const GraphNode& node, const std::string& type) {
        return env.Exploitability(std::vector<GraphNode>(env.NumPlayers(), node), type);
    }, py::arg("strategy"), py::arg("type_name") = "default");
    cls.def("utility", &CppEnvironment::Utility, py::arg("strategy"), py::arg("type_name") = "default");
    cls.def("utility", [](CppEnvironment& env, const GraphNode& node, const std::string& type) {
        return env.Utility(std::vector<GraphNode>(env.NumPlayers(), node), type);
    }, py::arg("strategy"), py::arg("type_name") = "default");
    cls.def("evaluate", &CppEnvironment::Metrics, py::arg("strategy"), py::arg("type_name") = "default");
    cls.def("evaluate", [](CppEnvironment& env, const GraphNode& node, const std::string& type) {
        return env.Metrics(std::vector<GraphNode>(env.NumPlayers(), node), type);
    }, py::arg("strategy"), py::arg("type_name") = "default");
}
