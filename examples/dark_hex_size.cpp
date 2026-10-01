// Exact size census for classical reveal-nothing Dark Hex, without a game file.
// Reference (game rules): OpenSpiel Dark Hex, classical reveal-nothing variant.
// https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/dark_hex/dark_hex.cc
// Build: c++ -O3 -std=c++17 examples/dark_hex_size.cpp -o /tmp/dark_hex_size
// Run:   /tmp/dark_hex_size 3
// Output is JSON. Board sizes 2 and 3 are supported.
//
// Node count: memoize the number of complete-history suffixes by physical
// board, each player's failed-probe mask, and acting player. These determine
// legal actions, transitions, and termination, even though they do not identify
// a perfect-recall information set. Every history still contributes separately
// to the count; memoization only reuses its suffix count.
//
// Information-set count: for each target player, retain the ordered sequence
// of its own attempts and success/failure results. The other player's failed
// probes can be omitted for this REACHABILITY census: they reveal nothing to
// the target, change neither stones nor turn, and cannot win the game. After
// any such probes, every actually empty cell remains a legal successful move.
// Conversely, an opponent may choose any such successful move directly.
// Therefore removing these probes preserves exactly the target information
// sets reachable at target decision points. Own probes are never removed.
// This reduction is not a replacement for gameplay, strategy evaluation, or
// best-response calculation: opponent policies can depend on failed probes.

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class DarkHexCensus {
public:
    explicit DarkHexCensus(int size)
        : size_(size), cells_(size * size), full_((1u << cells_) - 1) {
        std::array<unsigned, 9> neighbours{};
        std::array<unsigned, 2> start{}, finish{};
        constexpr int offsets[6][2] = {{-1, 0}, {-1, 1}, {0, -1},
                                       {0, 1}, {1, -1}, {1, 0}};
        for (int cell = 0; cell < cells_; ++cell) {
            const int row = cell / size_, col = cell % size_;
            const unsigned bit = 1u << cell;
            if (row == 0) start[0] |= bit;
            if (row == size_ - 1) finish[0] |= bit;
            if (col == 0) start[1] |= bit;
            if (col == size_ - 1) finish[1] |= bit;
            for (const auto& offset : offsets) {
                const int r = row + offset[0], c = col + offset[1];
                if (r >= 0 && r < size_ && c >= 0 && c < size_)
                    neighbours[cell] |= 1u << (r * size_ + c);
            }
        }
        for (int player = 0; player < 2; ++player) {
            wins_[player].resize(full_ + 1);
            for (unsigned mask = 0; mask <= full_; ++mask) {
                unsigned reached = mask & start[player], previous;
                do {
                    previous = reached;
                    for (int cell = 0; cell < cells_; ++cell)
                        if (previous & (1u << cell)) reached |= mask & neighbours[cell];
                } while (reached != previous);
                wins_[player][mask] = (reached & finish[player]) != 0;
            }
        }
    }

    void Print() {
        const auto nodes = CountNodes(0, 0, 0, 0, 0);
        std::array<std::uint64_t, 2> infosets{};
        for (int target = 0; target < 2; ++target) {
            infosets_.clear();
            CollectInfosets(0, 0, 0, 0, target, 0);
            infosets[target] = infosets_.size();
        }
        std::cout << "{\n"
                  << "  \"board_size\": " << size_ << ",\n"
                  << "  \"nodes\": " << nodes << ",\n"
                  << "  \"infosets_per_player\": [" << infosets[0] << ", " << infosets[1] << "],\n"
                  << "  \"information_sets\": " << infosets[0] + infosets[1] << ",\n"
                  << "  \"physical_states_memoized\": " << node_counts_.size() << ",\n"
                  << "  \"game_file_written\": false\n"
                  << "}\n";
    }

private:
    std::uint64_t CountNodes(unsigned black, unsigned white, unsigned failed_black,
                             unsigned failed_white, int player) {
        if (wins_[0][black] || wins_[1][white]) return 1;
        const std::uint64_t key = black | (std::uint64_t(white) << cells_) |
            (std::uint64_t(failed_black) << (2 * cells_)) |
            (std::uint64_t(failed_white) << (3 * cells_)) |
            (std::uint64_t(player) << (4 * cells_));
        const auto found = node_counts_.find(key);
        if (found != node_counts_.end()) return found->second;
        std::uint64_t total = 1;
        const unsigned legal = full_ & ~(player ? white | failed_white : black | failed_black);
        for (int action = 0; action < cells_; ++action) {
            const unsigned bit = 1u << action;
            if (!(legal & bit)) continue;
            if (player == 0)
                total += (white & bit)
                    ? CountNodes(black, white, failed_black | bit, failed_white, 0)
                    : CountNodes(black | bit, white, failed_black, failed_white, 1);
            else
                total += (black & bit)
                    ? CountNodes(black, white, failed_black, failed_white | bit, 1)
                    : CountNodes(black, white | bit, failed_black, failed_white, 0);
        }
        node_counts_.emplace(key, total);
        return total;
    }

    void CollectInfosets(unsigned black, unsigned white, unsigned attempted,
                         int player, int target, std::uint64_t sequence) {
        if (wins_[0][black] || wins_[1][white]) return;
        if (player == target) {
            infosets_.insert(sequence);
            const unsigned legal = full_ & ~attempted;
            for (int action = 0; action < cells_; ++action) {
                const unsigned bit = 1u << action;
                if (!(legal & bit)) continue;
                const bool collision = ((player == 0 ? white : black) & bit) != 0;
                // Nonzero base-(2*cells+1) digits encode both action and result.
                // At most 9 attempts fit safely in uint64_t for these boards.
                const auto next = sequence * (2 * cells_ + 1) + 1 + action +
                                  (collision ? cells_ : 0);
                if (collision)
                    CollectInfosets(black, white, attempted | bit, player, target, next);
                else
                    CollectInfosets(player == 0 ? black | bit : black,
                                    player == 1 ? white | bit : white,
                                    attempted | bit, 1 - player, target, next);
            }
        } else {
            const unsigned empty = full_ & ~(black | white);
            for (int action = 0; action < cells_; ++action) {
                const unsigned bit = 1u << action;
                if (empty & bit)
                    CollectInfosets(player == 0 ? black | bit : black,
                                    player == 1 ? white | bit : white,
                                    attempted, 1 - player, target, sequence);
            }
        }
    }

    int size_, cells_;
    unsigned full_;
    std::array<std::vector<bool>, 2> wins_;
    std::unordered_map<std::uint64_t, std::uint64_t> node_counts_;
    std::unordered_set<std::uint64_t> infosets_;
};

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "2" && std::string(argv[1]) != "3")) {
        std::cerr << "Usage: dark_hex_size [2|3]\n";
        return 1;
    }
    DarkHexCensus(argc == 2 ? std::stoi(argv[1]) : 3).Print();
}
