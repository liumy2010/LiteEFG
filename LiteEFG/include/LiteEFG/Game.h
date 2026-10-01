#ifndef LITEEFG_GAME_PLUGIN_H
#define LITEEFG_GAME_PLUGIN_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace liteefg {

// Players are 1..NumPlayers(), chance is 0, and terminals return -1.
// InformationSet identifies the acting player's complete information history.
// All states sharing a key must have the same ordered legal actions and the
// same preceding own (information set, action) sequence (perfect recall).
class State {
public:
    virtual ~State() = default;
    virtual int CurrentPlayer() const = 0;
    virtual std::vector<int> LegalActions() const = 0;
    virtual void ApplyAction(int action) = 0;
    virtual void UndoAction() = 0;
    virtual std::string InformationSet() const = 0;
    virtual std::vector<double> Returns() const = 0;
    virtual std::vector<std::pair<int, double>> ChanceOutcomes() const {
        throw std::logic_error("This game does not enumerate chance outcomes");
    }
    // Override for large chance spaces. Exact evaluation still requires
    // ChanceOutcomes. The input is uniformly distributed in [0, 1).
    virtual int SampleChance(double uniform) const {
        auto outcomes = ChanceOutcomes();
        if (outcomes.empty()) throw std::logic_error("Empty chance distribution");
        double sum = 0;
        for (const auto& item : outcomes) {
            sum += item.second;
            if (uniform < sum) return item.first;
        }
        return outcomes.back().first;
    }
    virtual double ChanceProbability(int action) const {
        for (const auto& item : ChanceOutcomes())
            if (item.first == action) return item.second;
        throw std::invalid_argument("Unknown chance action");
    }
};

class Game {
public:
    virtual ~Game() = default;
    virtual int NumPlayers() const = 0;
    virtual std::unique_ptr<State> NewInitialState() const = 0;
    virtual std::string Name() const = 0;
};

constexpr int kGameApiVersion = 1;
} // namespace liteefg

// Factory accepts a UTF-8 JSON object. Compile the plugin with the same C++
// standard library/ABI as LiteEFG. Objects are destroyed before dlclose.
#if defined(_WIN32)
#define LITEEFG_GAME_EXPORT extern "C" __declspec(dllexport)
#else
#define LITEEFG_GAME_EXPORT extern "C" __attribute__((visibility("default")))
#endif
#define LITEEFG_REGISTER_GAME(GameClass) \
    LITEEFG_GAME_EXPORT int liteefg_game_api_version() { \
        return ::liteefg::kGameApiVersion; \
    } \
    LITEEFG_GAME_EXPORT ::liteefg::Game* liteefg_create_game(const char* parameters) { \
        return new GameClass(std::string(parameters)); \
    } \
    LITEEFG_GAME_EXPORT void liteefg_destroy_game(::liteefg::Game* game) { delete game; }

#endif
