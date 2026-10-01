# Reference output checks

The active references capture the workspace engine's outputs. Each reference
identifies the engine, baseline sources, and runtime used to generate it.

The active reference output check platform is Linux/libstdc++, including Ubuntu under Windows Subsystem for Linux (WSL).
Only Linux reference snapshots are checked in.

| Active reference on Linux | Engine used to record it | Baselines | Purpose |
| --- | --- | --- | --- |
| `baselines/reference-linux.json` | Workspace engine | Workspace | Default reference output check |
| `baselines/reference-arguments-linux.json` | Workspace engine | Workspace | Algorithm argument combinations |
| `baselines/reference-public-baselines-linux.json` | Workspace engine | Frozen public | Engine reference output check with fixed baseline sources |

The frozen public baselines come from
[public LiteEFG commit 80913416964fb8f5aa9166248bd6f32b9ae32ee1](https://github.com/liumy2010/LiteEFG/tree/80913416964fb8f5aa9166248bd6f32b9ae32ee1/LiteEFG/baselines).
The Python sources, their MIT license, commit and 256-bit Secure Hash Algorithm (SHA-256) hashes are stored in
`baselines/public/`.

## Benchmarks for review

All games have two players and zero-sum utilities. Sizes below count the complete
tree after conversion of simultaneous actions to turn-based actions, using
OpenSpiel 1.5.

| Identifier | Exact OpenSpiel game | Nodes | Information sets | Purpose |
| --- | --- | ---: | ---: | --- |
| kuhn | `kuhn_poker(players=2)` | 58 | 12 | Small imperfect-information poker with chance |
| leduc | `leduc_poker(players=2,suit_isomorphism=True)` | 1,939 | 288 | Deeper poker, chance, more information sets |
| goofspiel | `goofspiel(num_cards=3,players=2,imp_info=True,points_order=descending)` | 67 | 16 | Simultaneous actions converted to sequential form |

`baselines/manifest.json` is the reviewable default-parameter specification. Each of the 15
baseline modules runs with its Python constructor defaults, which are also
captured in the reference so a default change fails comparison. Workspace
constructor and command-line defaults agree. Each layer uses the defaults of its
baseline sources. Outcome traversal is
used for Balanced Follow the Regularized Leader (Balanced FTRL),
Balanced Online Mirror Descent (Balanced OMD), Implicit Exploration Online
Mirror Descent (IXOMD), and Outcome-Sampling Monte Carlo Counterfactual Regret
Minimization (OS-MCCFR); the others use Enumerate.
One extra Counterfactual Regret Minimization (CFR) case uses External traversal: **16 configurations × 3 games = 48
cases**. Adding or deleting an algorithm without updating the manifest fails the
coverage check. Traversals are specified per configuration; this is not the full
cross product of every algorithm with every traversal mode.

Each case performs 50 calls to `graph.update_graph(env)`, followed by
`env.update_strategy(graph.current_strategy())`. Checkpoints are after exactly
1, 10, 25 and 50 calls. Clairvoyant Mirror Descent (CMD) therefore performs 50 inner updates (5 outer updates
with its default `inner_epoch=10`). The harness calls each implementation's
default `current_strategy()` selector. This is a short reference output
check, not a convergence-quality claim or exhaustive coverage of every
algorithm hyperparameter.

At each checkpoint we measure all of `last-iterate`, `avg-iterate`, and
`linear-avg-iterate`:

- `utility`: each player's expected payoff;
- `deviation_gain`: each player's best-response gain, from `env.exploitability`;
- `nash_conv`: the sum of those gains (printed as `Exploitability` by the baseline runner);
- `exploitability`: half the sum of those gains, using the two-player zero-sum convention;
- `information_sets`: the number of information sets per player.

Each layer therefore compares **48 cases × 4 checkpoints × 3 strategy selectors
= 576 metric snapshots**. A case count does not count each selector separately.

The runner also rejects non-finite metrics, negative deviation gains, non-zero-sum
utilities and invalid strategy probabilities. Numerical comparison uses
`abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected)`. Missing/extra cases,
checkpoints, metrics, configuration changes and constructor-default changes fail.
The thresholds are for floating-point drift, not statistical confidence intervals.

## Algorithm argument combinations

`baselines/manifest-arguments.json` defines a workspace-only reference output check suite with
**115 configurations × 3 games = 345 individually parametrized pytest cases**.
Each configuration uses the same 50 updates, four checkpoints and three strategy
selectors as the default suite: **4,140 metric snapshots**. All finite algorithm
switches are crossed within each baseline; changing just one switch at a time
would miss interactions between branches.

| Baseline | Arguments crossed | Configurations |
| --- | --- | ---: |
| CMD | `regularizer` × `weighted` × `inner_epoch={1,10}` | 8 |
| Dilated Optimistic Mirror Descent (DOMD) | `regularizer` × `weighted` | 4 |
| Discounted Counterfactual Regret Minimization (DCFR) | `alpha={-11,1.5,11}` × `beta={-11,0,11}` | 9 |
| Follow the Perturbed Leader (FTPL) | `noise_type={exponential,uniform,normal}` | 3 |
| Magnetic Mirror Descent (MMD), Q-Function based Regret Minimization (QFR) | `regularizer` × `feedback={Q,traj-Q,counterfactual,Outcome}` × `weighted` | 16 each |
| `OS_MCCFR` | `rm_plus` × `balanced` | 4 |
| Regularized Counterfactual Regret Minimization (Reg-CFR), Regularized Dilated Optimistic Mirror Descent (Reg-DOMD) | `regularizer` × `weighted` × `out_reg` × `shrink_iter={1,10,100000}` | 24 each |
| CFR | Default plus External traversal | 2 |
| `Balanced_FTRL`, `Balanced_OMD`, Counterfactual Regret Minimization+ (`CFRplus`), `IXOMD`, Predictive Counterfactual Regret Minimization+ (`PCFR`) | Constructor defaults | 1 each |

Every boolean includes both values; `regularizer` includes Entropy and Euclidean.
MMD/QFR Outcome feedback uses Outcome traversal; their other feedback modes use
Enumerate. Traversal otherwise follows the default suite. The matrix does not
cross every traversal mode with every algorithm.

The numeric representatives exercise different code paths: DCFR exponents below,
inside and above `[-10,10]`; CMD publishing after one or ten inner updates; and
regularization shrinking every update, periodically, or after the test horizon.
Independent tests check DCFR's exact ±10 boundaries and native update schedules,
including Reg_CFR's first-update exception. Continuous coefficients such as
`eta`, `tau`, `gamma`, `delta` and `kappa` keep their constructor defaults; this is
branch-combination coverage, not a grid over all hyperparameter values.

`test_baseline_arguments.py` also checks the complete Cartesian products against
an independently declared coverage specification and the supported constructor
and command-line interface (CLI) switches, so a missing combination or newly added switch fails coverage.
It also verifies that every runner passes the same algorithm defaults as its
constructor, including the enabled-by-default OS_MCCFR balanced exploration flag.
Each case compares numeric outputs against
`baselines/reference-arguments-linux.json`, which uses the workspace engine and
workspace baselines. All 48 default-suite cases are also present in the argument
suite, with identical outputs in both active Linux references.
Argument-suite artifacts are saved under
`artifacts/reference-output-checks/arguments/`.

Run only the argument suite and independent schedule checks:

```sh
python -m pytest tests/test_baseline_arguments.py tests/test_baseline_schedules.py
```

The reference output checker can also check this suite directly:

```sh
python tests/reference_output_check.py check \
  --manifest tests/baselines/manifest-arguments.json \
  --reference tests/baselines/reference-arguments-linux.json \
  --artifacts artifacts/reference-output-checks/arguments
```

For an explicitly requested update, use the same review and independent-test
requirements as the default references, then record both runs with:

```sh
python tests/reference_output_check.py record-workspace --overwrite \
  --manifest tests/baselines/manifest-arguments.json \
  --reference tests/baselines/reference-arguments-linux.json \
  --artifacts artifacts/reference-output-checks/record-arguments
```

## Run after modifying code

Use Python 3.12. Install the pinned dependencies and compile the checkout:

```sh
python -m pip install -r tests/requirements.txt
python setup.py build_ext --inplace
python -m pytest tests
```

On Windows, run these commands inside WSL (Ubuntu) using a Linux Python 3.12
environment and Linux dependencies. The active references and continuous integration (CI) use Linux.
If a Conda `readline` installation causes pytest's capture initialization to crash,
use `python -m pytest -p no:capture tests`.

### Documentation examples

The documentation tests read the actual fenced code from `docs/`, rather than
maintaining copies. Run them separately with:

```sh
python -m pytest tests/test_docs_*.py
```

They execute all 52 Python blocks with the context specified by each page, all
five baseline CLI recipes at 10,000 iterations each, the quick-start
script command, and the import check. Assertions cover numerical results,
probability normalization, OpenSpiel agreement, comma-separated values (CSV) round-trips, cache
regeneration, and native graph update behavior. Terminal interactions run their
specified number of games with scripted legal input. Generated games and exports
stay in temporary directories; the tests do not change `HOME` or use the user's
game cache. A page/language inventory rejects added code blocks until their
coverage is reviewed.

All shell fences receive a syntax check. Dependency installation, Git checkout,
and long-running website servers are setup steps rather than pytest workloads;
validate those separately when changing them. The PowerShell installation recipe
requires Windows validation. Mermaid is rendered by the separate documentation
build, and the game-format excerpts are verified against the complete bundled
game. These tests also run as part of `python -m pytest tests` in the
Reference output checks workflow.

### Baseline comparisons

Pytest checks both layers against their corresponding saved answers:

1. Frozen public baselines on the workspace engine against
   `reference-public-baselines-linux.json`, to isolate engine changes.
2. Workspace baselines on the workspace engine against `reference-linux.json`,
   to also detect algorithm changes.

Each layer writes `actual.json` and `comparison.json` under
`artifacts/reference-output-checks/{public,workspace}/`. Failures identify the
baseline/game, iteration, strategy type and metric, with expected/actual values,
absolute difference and allowed tolerance. A stale C++ extension is rejected;
rebuild it before comparing. For just one layer:

```sh
python tests/reference_output_check.py check
python tests/reference_output_check.py check --baseline-source public
```

### Checkpoint continuation

`test_checkpoint_baselines.py` repeats every baseline/game case with
`graph.save(path)` followed by `graph.load(path)` after updates 10, 20, 30, 40,
and 50. Training performs exactly 50 updates, and measurements on a reload
iteration use the restored state. These **441 individually parametrized cases**
cover the 48 workspace defaults, 48 frozen-public defaults, and 345 workspace
argument combinations, including stochastic traversals, all FTPL noise types,
CMD inner epochs, and regularization schedules.

The checkpoint runs use the references, measurement checkpoints, strategy
selectors, and tolerances of the corresponding uninterrupted suites. They never
record or replace a reference. Results are written under
`artifacts/reference-output-checks/checkpoints/{workspace,public,arguments}/`.

```sh
python -m pytest -p no:capture tests/test_checkpoint_baselines.py
python tests/reference_output_check.py check --checkpoint-interval 10 \
  --artifacts artifacts/reference-output-checks/checkpoints/workspace
python tests/reference_output_check.py check --baseline-source public \
  --checkpoint-interval 10 \
  --artifacts artifacts/reference-output-checks/checkpoints/public
```

`--checkpoint-interval` is a check-only option for the workspace engine; the
recording commands use an uninterrupted, two-run verification workflow.
Pipeline tests also verify all five save/load calls, their placement before
measurements, and the 50-update horizon.

## Parallel execution

`test_parallel.py` runs all 345 workspace argument cases and 48 frozen-public
cases with 1, 2, and 4 native threads. Serial outputs are checked against the
Linux references; parallel outputs must match serial outputs exactly,
including every strategy probability at each checkpoint and the final random number generator (RNG)
state. No reference is regenerated. Results and engine/baseline provenance are
written to `artifacts/reference-output-checks/parallel/{arguments,public}/actual.json`.

Focused tests cover forward/backward dependencies, update colors and players,
all traversal modes, changing thread counts, checkpoint continuation, irregular
history depths, ordered aggregation, and random dimensions. Opponent and
non-tree own-player dependencies execute serially. The Linux CI
test command includes these tests, even on machines with fewer CPUs than the
requested thread count; correctness does not depend on a speedup.

```sh
python -m pytest -p no:capture tests/test_parallel.py tests/test_parallel_edge_cases.py
```

## Independent correctness checks and CI

Independent tests cover DCFR/OpenSpiel agreement, Reg-DOMD sequence-form
optimality, projection optimality and idempotence, balanced weights, native
operations, traversal, and the OpenSpiel adapter. They check correctness
separately from the reference snapshots.

GitHub Actions runs the build and tests on every push and pull request, and can be
triggered manually. It uploads the output artifacts even on failure. Local edits
are checked by the commands above; CI checks them when pushed.

## Record an explicitly approved workspace reference

After reviewing intended changes, rebuild and pass the independent correctness
tests. Then run the explicit recording command for each layer:

```sh
python setup.py build_ext --inplace
python tests/reference_output_check.py record-workspace --overwrite \
  --artifacts artifacts/reference-output-checks/record-workspace
python tests/reference_output_check.py record-workspace --baseline-source public \
  --overwrite --artifacts artifacts/reference-output-checks/record-public-baselines
python -m pytest tests
```

`record-workspace` uses the freshly built workspace engine and the selected
baseline layer. It runs all cases twice in fresh processes and refuses to replace
the reference if either run fails, produces invalid/incomplete output, or differs
beyond the existing tolerance. Existing files require `--overwrite`; `check`
never creates or updates a reference. `--oracle-root` is rejected for workspace
recording. Reference provenance retains engine/source hashes and the runtime;
these identify the recorded implementation without requiring future code changes
to preserve those hashes.

To review a candidate without replacing a checked-in file, add
`--reference /tmp/liteefg-reference-candidate.json`. Both recording runs are saved
under the specified artifacts directory.

## Record a public-release reference

The `record` command requires an isolated PyPI installation outside this checkout,
validates both Python-package and native-extension import locations, and loads the
frozen public baseline sources. Without `--reference`, its destination is
`baselines/reference-pypi-<platform>.json`. It rejects an explicit
`--baseline-source workspace`.

```sh
# Use the pinned dependencies above with the selected public release.
python -m pip install --target /tmp/liteefg-public-oracle --no-deps LiteEFG==0.1.5
python tests/reference_output_check.py record \
  --oracle-root /tmp/liteefg-public-oracle \
  --reference /tmp/liteefg-reference-candidate.json
```

For a deliberate upgrade to the newest PyPI release, use a **fresh** target
directory and `pip install LiteEFG` instead of the version pin. Updating the frozen
public baselines requires reviewing the public commit, copying its sources and
license, and updating `public/source.json`. Review the resulting output diff before
replacing the checked-in reference. Replacing an existing reference requires
`record --overwrite`; an ordinary test/check never creates or rewrites references.

Public recording also runs every case twice in fresh processes and refuses to save if either
run fails or their results differ beyond tolerance. The reference includes the
PyPI distribution version, public GitHub commit, source and extension hashes,
dependency versions, seed, parameters, traversal modes and checkpoints. The two
recording runs are retained under `artifacts/reference-output-checks/` for inspection.

## Reproducibility limits

Each case starts a fresh process and seeds Python, NumPy and LiteEFG with
`0`. Game files are regenerated in a temporary directory: the runner does
not reuse `~/game_instances` or change `HOME`. Process isolation also prevents C++
random-distribution caches from leaking between cases.

The active Linux references and CI use Linux/libstdc++ and pinned OpenSpiel,
NumPy and pandas versions.
LiteEFG uses `std::default_random_engine` and standard library distributions,
whose streams are not portable across C++ libraries. Other platforms require
separately reviewed references recorded with the appropriate engine and baseline
sources. The runner fails explicitly when a selected reference is missing or its
constructor defaults do not match the selected baseline sources.
Changing the game library or runtime version also requires an explicit reference
review. Do not relax tolerances to hide a change in stochastic trajectories.
