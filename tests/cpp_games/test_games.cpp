#include <LiteEFG/Game.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {

enum class Mode { Hidden, Matrix, Sequential, NormalForm, InvalidActions,
                  InvalidRecall, DeepChance, NoDecision };

class TestState final : public liteefg::State {
public:
    explicit TestState(Mode mode) : mode_(mode) {}

    int CurrentPlayer() const override {
        const auto depth = history_.size();
        if (mode_ == Mode::NoDecision) return depth == 0 ? 0 : -1;
        if (mode_ == Mode::NormalForm) return depth == 3 ? -1 : int(depth) + 1;
        if (mode_ == Mode::DeepChance) return depth < 16 ? 0 : (depth == 16 ? 1 : -1);
        if (mode_ == Mode::Sequential || mode_ == Mode::InvalidRecall)
            return depth == 3 ? -1 : (depth == 1 ? 0 : 1);
        // FileEnv inserts a deterministic chance root. Matching it here permits
        // seed-for-seed independent comparisons of the sampled traversals.
        if (mode_ == Mode::Matrix) return depth == 4 ? -1 : std::max(0, int(depth) - 1);
        return depth == 2 ? -1 : int(depth);
    }

    std::vector<int> LegalActions() const override {
        const int player = CurrentPlayer();
        if (player == -1) return {};
        if (mode_ == Mode::Matrix && history_.empty()) return {99};
        if (player == 0) return {2, 5};
        if (mode_ == Mode::NormalForm && player == 2) return {10, 20, 30};
        if (mode_ == Mode::InvalidActions && history_[0] == 5) return {10, 30};
        return {10, 20};
    }

    std::vector<std::pair<int, double>> ChanceOutcomes() const override {
        if (mode_ == Mode::Matrix && history_.empty()) return {{99, 1.0}};
        if (mode_ == Mode::DeepChance) return {{2, 0.5}, {5, 0.5}};
        return {{2, 0.25}, {5, 0.75}};
    }

    void ApplyAction(int action) override {
        const auto legal = LegalActions();
        if (std::find(legal.begin(), legal.end(), action) == legal.end())
            throw std::invalid_argument("Illegal fixture action");
        history_.push_back(action);
    }

    void UndoAction() override {
        if (history_.empty()) throw std::logic_error("Undo at initial state");
        history_.pop_back();
    }

    std::string InformationSet() const override {
        if (mode_ == Mode::NormalForm || mode_ == Mode::Matrix)
            return "p" + std::to_string(CurrentPlayer());
        if (mode_ == Mode::Sequential || mode_ == Mode::InvalidRecall) {
            if (history_.empty()) return "root";
            if (mode_ == Mode::InvalidRecall) return "forgot";
            return history_[0] == 10 ? "left" : "right";
        }
        return "guess";
    }

    std::vector<double> Returns() const override {
        if (CurrentPlayer() != -1) throw std::logic_error("Returns before terminal");
        if (mode_ == Mode::NoDecision)
            return history_[0] == 2 ? std::vector<double>{-2.0, -7.0} : std::vector<double>{-4.0, -3.0};
        if (mode_ == Mode::NormalForm) {
            const double a = history_[0] / 10 - 1;
            const double b = history_[1] / 10 - 1;
            const double c = history_[2] / 10 - 1;
            return {2*a - b + 3*c - 4*a*c + a*b,
                    -a + 2*b - c + a*b - 3*b*c,
                    a - 2*b + c + 2*a*c + b*c};
        }
        double value;
        if (mode_ == Mode::Matrix || mode_ == Mode::Sequential ||
            mode_ == Mode::InvalidRecall) {
            constexpr double payoff[2][2][2] = {{{4, -2}, {6, 0}}, {{2, 8}, {-4, 10}}};
            const bool sequential = mode_ != Mode::Matrix;
            const int chance = history_[1] == 5;
            const int first = history_[sequential ? 0 : 2] == 20;
            const int last = history_[sequential ? 2 : 3] == 20;
            value = payoff[chance][first][last];
        } else {
            const int chance = mode_ == Mode::DeepChance ? history_[15] : history_[0];
            value = ((chance == 5) == (history_.back() == 20)) ? 1.0 : -1.0;
        }
        return {value, -value};
    }

private:
    Mode mode_;
    std::vector<int> history_;
};

class TestGame final : public liteefg::Game {
public:
    explicit TestGame(const std::string& parameters) {
        if (parameters.find("normal_form3") != std::string::npos) mode_ = Mode::NormalForm;
        else if (parameters.find("no_decision") != std::string::npos) mode_ = Mode::NoDecision;
        else if (parameters.find("invalid_actions") != std::string::npos) mode_ = Mode::InvalidActions;
        else if (parameters.find("invalid_recall") != std::string::npos) mode_ = Mode::InvalidRecall;
        else if (parameters.find("deep_chance") != std::string::npos) mode_ = Mode::DeepChance;
        else if (parameters.find("sequential") != std::string::npos) mode_ = Mode::Sequential;
        else if (parameters.find("matrix") != std::string::npos) mode_ = Mode::Matrix;
    }

    int NumPlayers() const override { return mode_ == Mode::NormalForm ? 3 : 2; }
    std::unique_ptr<liteefg::State> NewInitialState() const override {
        return std::make_unique<TestState>(mode_);
    }
    std::string Name() const override { return "independent-test-game"; }

private:
    Mode mode_ = Mode::Hidden;
};

} // namespace

LITEEFG_REGISTER_GAME(TestGame)
