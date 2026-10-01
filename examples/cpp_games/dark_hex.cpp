// Classical Dark Hex with private failed attempts and perfect recall.
// Reference (game rules): OpenSpiel Dark Hex, classical reveal-nothing variant.
// https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/dark_hex/dark_hex.cc
// The information key records the player's ordered attempts and their outcomes.
// Opponent unsuccessful attempts reveal neither their number nor timing.
#include <LiteEFG/Game.h>

#include <array>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class DarkHexGame;

class DarkHexState final : public liteefg::State {
public:
    explicit DarkHexState(const DarkHexGame& game) : game_(game) {}

    int CurrentPlayer() const override { return winner_ ? -1 : player_ + 1; }
    std::vector<int> LegalActions() const override;
    void ApplyAction(int action) override;
    void UndoAction() override;
    std::string InformationSet() const override {
        if (winner_) throw std::logic_error("Terminal Dark Hex state has no information set");
        std::string result(1, static_cast<char>('1' + player_));
        result.append(attempts_[player_].data(), attempt_count_[player_]);
        return result;
    }
    std::vector<double> Returns() const override {
        const double value = winner_ == 1 ? 1.0 : (winner_ == 2 ? -1.0 : 0.0);
        return {value, -value};
    }

private:
    struct Undo {
        std::uint16_t bit;
        std::uint8_t player;
        bool success;
    };
    const DarkHexGame& game_;
    std::array<std::uint16_t, 2> stones_{};
    std::array<std::uint16_t, 2> attempted_{};
    std::array<std::array<char, 16>, 2> attempts_{};
    std::array<int, 2> attempt_count_{};
    std::array<Undo, 32> undo_{};
    int depth_ = 0;
    int player_ = 0;
    int winner_ = 0;
};

class DarkHexGame final : public liteefg::Game {
public:
    explicit DarkHexGame(const std::string& parameters) : size_(ParseSize(parameters)) {
        const int cells = size_ * size_;
        std::array<std::uint16_t, 16> neighbours{};
        std::array<std::uint16_t, 2> start{}, finish{};
        constexpr int offsets[6][2] = {{-1, 0}, {-1, 1}, {0, -1},
                                       {0, 1}, {1, -1}, {1, 0}};
        for (int cell = 0; cell < cells; ++cell) {
            const int row = cell / size_, col = cell % size_;
            const std::uint16_t bit = static_cast<std::uint16_t>(1u << cell);
            if (row == 0) start[0] |= bit;
            if (row == size_ - 1) finish[0] |= bit;
            if (col == 0) start[1] |= bit;
            if (col == size_ - 1) finish[1] |= bit;
            for (const auto& offset : offsets) {
                const int r = row + offset[0], c = col + offset[1];
                if (r >= 0 && r < size_ && c >= 0 && c < size_)
                    neighbours[cell] |= static_cast<std::uint16_t>(1u << (r * size_ + c));
            }
        }
        // A board-mask lookup makes both forward and reverse traversal cheap.
        // These tables have 512 entries per player on the 3x3 board.
        for (int player = 0; player < 2; ++player) {
            wins_[player].resize(1u << cells);
            for (unsigned mask = 0; mask < (1u << cells); ++mask) {
                unsigned reached = mask & start[player], previous;
                do {
                    previous = reached;
                    for (int cell = 0; cell < cells; ++cell)
                        if (previous & (1u << cell)) reached |= mask & neighbours[cell];
                } while (reached != previous);
                wins_[player][mask] = (reached & finish[player]) != 0;
            }
        }
    }

    int NumPlayers() const override { return 2; }
    std::unique_ptr<liteefg::State> NewInitialState() const override {
        return std::make_unique<DarkHexState>(*this);
    }
    std::string Name() const override {
        return "dark_hex(board_size=" + std::to_string(size_) +
               ",gameversion=cdh,obstype=reveal-nothing,perfect_recall=true)";
    }
    int Cells() const { return size_ * size_; }
    bool Wins(int player, std::uint16_t stones) const { return wins_[player][stones]; }

private:
    static int ParseSize(const std::string& parameters) {
        if (std::regex_match(parameters, std::regex(R"(\s*(\{\s*\})?\s*)"))) return 3;
        // The plugin factory receives a JSON object. Bare sizes are useful when
        // loading the plugin directly from C++ diagnostic programs.
        std::smatch match;
        if (std::regex_match(parameters, match,
                std::regex(R"(\s*\{\s*"board_size"\s*:\s*([234])\s*\}\s*)")) ||
            std::regex_match(parameters, match, std::regex(R"(\s*([234])\s*)")))
            return std::stoi(match[1].str());
        throw std::invalid_argument("Dark Hex parameters must be {\"board_size\": 2, 3, or 4}");
    }

    int size_;
    std::array<std::vector<std::uint8_t>, 2> wins_;
};

std::vector<int> DarkHexState::LegalActions() const {
    std::vector<int> actions;
    if (winner_) return actions;
    actions.reserve(game_.Cells() - attempt_count_[player_]);
    for (int cell = 0; cell < game_.Cells(); ++cell)
        if (!(attempted_[player_] & (1u << cell))) actions.push_back(cell);
    return actions;
}

void DarkHexState::ApplyAction(int action) {
    if (winner_ || action < 0 || action >= game_.Cells() ||
        (attempted_[player_] & (1u << action)))
        throw std::invalid_argument("Illegal Dark Hex action");
    const auto bit = static_cast<std::uint16_t>(1u << action);
    const bool success = !(stones_[1 - player_] & bit);
    undo_[depth_++] = {bit, static_cast<std::uint8_t>(player_), success};
    attempted_[player_] |= bit;
    // Uppercase marks a collision; lowercase marks a successful placement.
    attempts_[player_][attempt_count_[player_]++] =
        static_cast<char>((success ? 'a' : 'A') + action);
    if (success) {
        stones_[player_] |= bit;
        if (game_.Wins(player_, stones_[player_])) winner_ = player_ + 1;
        player_ = 1 - player_;
    }
}

void DarkHexState::UndoAction() {
    if (!depth_) throw std::logic_error("Cannot undo the initial Dark Hex state");
    const Undo& previous = undo_[--depth_];
    player_ = previous.player;
    winner_ = 0;
    attempted_[player_] ^= previous.bit;
    --attempt_count_[player_];
    if (previous.success) stones_[player_] ^= previous.bit;
}

} // namespace

LITEEFG_REGISTER_GAME(DarkHexGame)
