---
description: Train policies for fog-of-war chess and understand its rules, actions, and model features.
---

# Dark chess

Dark chess hides the opponent's pieces outside each player's visible squares.
Use the [Proximal Policy Optimization (PPO) runner](https://github.com/liumy2010/LiteEFG/blob/main/examples/dark_chess_ppo.py)
for training, or the feature descriptions below to build your own networks.

## Rules and defaults

`leg.DarkChess(max_plies=1000, history_length=8)` implements the
[MIT Fog of War chess game](https://www.mit.edu/~6.7980/fow/?tab=rules) with pure
JAX operations. Player 0 controls White and player 1 controls Black. Players
alternate moves, with White moving first. Check does not restrict the legal
move list.

After each move, the game checks these conditions in order:

1. If the opponent can capture the mover's king next, the mover loses immediately.
   No additional king-capture move is executed.
2. Threefold repetition or 100 consecutive plies without a pawn move or capture
   produces a draw.
3. A player with no available moves loses.
4. Reaching `max_plies` produces a draw.

These rules follow the referee in the
[official starter package](https://www.mit.edu/~6.7980/fow/starter/starter.zip).
`max_plies` counts individual moves and is also the environment's `max_steps`;
the default of 1,000 matches the tournament limit.

Castling requires the appropriate rights, piece placement, and clear path;
attacked squares do not prevent castling. There is no insufficient-material
draw.

## Actions and board observations

The fixed action space contains 4,228 actions. Squares use `a1=0`, `b1=1`, through
`h8=63`. The first 4,096 actions encode `64 * source + destination`, including
queen promotions. A further 132 actions distinguish promotions to a knight,
bishop, or rook. The legal-action mask identifies available moves and is false
for the inactive player and after termination.

`env.action_from_uci("e2e4")`
converts a Universal Chess Interface (UCI) move to its action index; `env.action_to_uci(action, state)`
converts an index back using the state before the move, including promotion
suffixes.

`env.observation(state)` returns a `[2, 64]` array containing each player's
visible board. White pawn, knight, bishop, rook, queen, and king use values 1
through 6; Black's pieces use -1 through -6. Zero denotes a visible empty
square, and 7 denotes an unseen square.

A player sees their own pieces, their
available move destinations, and an opponent pawn that can be captured en
passant. A pawn does not reveal an opponent blocking its forward square, or
an empty diagonal, unless another piece can move to that square.
Both players use the same `a1` through `h8` square ordering.

## Model features

`history_length` is a positive integer specifying the number of observed board
frames, including the current board. Frames are newest first, and a new frame
is recorded after every individual move. Unfilled history frames are zero.
`features(state)` supplies these arrays:

| Feature | Shape for one game | Contents |
| --- | --- | --- |
| `information_set` | `[2, history_length * 896 + 1]` | Flattened observed board history followed by the player's color: White 1, Black 0 |
| `full_info` | `[2, (1 + 2 * history_length) * 896 + 2 * history_length + 73]` | Packed current-mover critic input, identical in both player rows |
| `history_mask` | `[2, history_length]` | Boolean mask identifying populated history frames |

### Board encoding

Each board uses 14 float32 planes in this order: own pawn, knight, bishop,
rook, queen, king; the opponent's corresponding six piece types; visible empty;
and unseen. Each real square has one active channel, so an unseen square is
distinct from both a visible empty square and zero padding.

The final two
dimensions are rank and file. Rank zero starts at the observing player's home
rank: Black's ranks are flipped, while files remain in `a` through `h` order.
The piece colors in these feature planes are relative to the observing player.

Action indices and UCI moves retain absolute board coordinates; undo the rank
flip when deriving an action from Black's tensor coordinates.

`env.board_features(state)` returns the observed history as
`[2, history_length, 14, 8, 8]`, and `env.full_board_features(state)` returns the
current complete board as `[2, 14, 8, 8]`. These structured helpers encode the
same boards as the flat feature arrays. Custom models can use these structured
boards or read the flat arrays directly. The example's Multilayer Perceptron (MLP) actor reads only
`information_set`, including the player's color and zero-padded history.
`history_mask` is available to custom graphs through `self.env.history_mask`.

### Critic feature layout

The critic's `full_info` vector begins with `1 + 2 * history_length` board
frames: the current true board, the current mover's observed history, and the
opponent's observed history. Histories are newest first. Every frame uses the
current mover's rank orientation and relative piece colors, including the
opponent's private observations. The opponent's unseen squares remain unseen
in that history; the separate true-board frame contains all current pieces.
The next `2 * history_length` entries are the mover's and opponent's history
masks, followed by 73 metadata values:

| Metadata entries | Contents |
| --- | --- |
| 0 | Current mover's color: White 1, Black 0 |
| 1–4 | Mover queenside/kingside castling rights, then opponent queenside/kingside rights |
| 5–69 | En passant target as a 65-way one-hot vector in mover coordinates; index 64 means none |
| 70 | Halfmove clock divided by 100 |
| 71 | Remaining plies divided by `max_plies`, clipped below at zero |
| 72 | Current position's repetition count divided by 3 |

`env.critic_board_features(state)`, `env.critic_history_mask(state)`, and
`env.critic_metadata(state)` expose the corresponding structured parts.
`env.full_info_size` gives the packed vector size. Castling rights, true board
state, and the opponent's private history are privileged critic inputs. They
are not supplied to the actor.

A finite board-history window is an observation
abstraction, not a perfect-recall information set. The critic metadata includes
the current repetition count rather than the complete repetition history, so
the packed input is not a complete Markov state either.

## MLP actors and shared critic

`LiteEFG.drl.models.MLP(output_size, hidden_size=128)` applies two fully connected
hidden layers with Rectified Linear Unit (ReLU) activations and a linear output layer. It reads the
flat feature vector directly and preserves leading batch dimensions. The
environment supplies zeros for unfilled history frames.

Use `MLP(output_size=env.num_actions)` for policy logits and `MLP(output_size=1)`
for the current mover's remaining return. Wrap the critic once with `leg.model`
and pass it to `PPO.graph(..., critic_feature="full_info")` so White and Black
share its parameters and optimizer state. Pass distinct actor resources in
`policy_network` for independent policies.

The executable [dark chess PPO example](https://github.com/liumy2010/LiteEFG/blob/main/examples/dark_chess_ppo.py)
uses these MLPs as simple examples. Each actor reads only its own observed
history and color; the shared critic reads the packed `full_info` vector.
`--hidden-size` controls the width of both hidden layers, with a default of 128.
Users can supply their own Flax modules to explore other architectures.
The example saves each actor's weights separately from the shared critic
weights and writes the run configuration
and metrics to `run.json`.

## Plan for rollout memory

The trainer retains feature arrays and action masks for every sampled ply,
including inactive turns and terminal padding. With the default 1,000-ply
horizon and eight-frame history, the features and legal-action masks require
about 188 MB per collected game before training caches. Start with an explicit
small `batch_size`, such as 1 or 4.

See [parallelism and memory](./scaling.md) for microbatching and packed rollouts.
