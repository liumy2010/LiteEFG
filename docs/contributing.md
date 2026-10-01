---
description: Preview, build, check, and deploy the LiteEFG VitePress documentation without compiling the Python package.
---

# Contributing to the docs

Documentation lives in `docs/` as Markdown and is built with VitePress. Building
it requires **Node.js 22.12 or newer** and npm; continuous integration (CI) uses Node.js 24.

## Preview locally

From the repository root:

```sh
npm ci
npm run docs:dev
```

Open `http://localhost:5173/LiteEFG/`. Markdown edits reload automatically. The production build uses the same base path:

```sh
npm run docs:build
npm run docs:check
npm run docs:preview
```

Open `http://localhost:4173/LiteEFG/`. VitePress may choose a different port if that one is occupied; use the address printed in the terminal.

Restart `docs:preview` after rebuilding: it indexes generated assets at startup.
Use `docs:dev` while editing so that page changes and navigation stay live.

The build fails on unresolved Markdown page links. `docs:check` then scans generated HyperText Markup Language (HTML) pages for missing internal pages, fragment targets, assets, and links that escape the `/LiteEFG/` base; it also checks local Cascading Style Sheets (CSS) assets. It does not fetch external websites or execute the Python examples. Review diagrams, search, and responsive layouts in the browser as well.

Check Python examples with `python -m pytest tests/test_docs_*.py`, after
installing the [test dependencies and building the native extension](https://github.com/liumy2010/LiteEFG/blob/main/tests/README.md#run-after-modifying-code).
These tests execute code extracted from Markdown. Installation commands and
website servers need separate validation; the website build does not execute
Python examples.

The npm lockfile pins the build dependencies. The `vite` override in `package.json` keeps VitePress 1.x’s development server on a patched Vite 6 release supported by its Vue plugin. Recheck that override when upgrading VitePress, and run the build, preview, and `npm audit` together.

## Where things live

| Path | Purpose |
| --- | --- |
| `docs/index.md` | Landing page |
| `docs/guide/` | Installation, quick start, concepts, graph, environments, algorithms, examples |
| `docs/guide/baselines/` | Individual tabular baseline guides |
| `docs/guide/environments/` | FileEnv, OpenSpiel, and CppEnv guides |
| `docs/guide/deep-learning/` | Deep Reinforcement Learning (DRL) baselines, models, training, evaluation, custom objectives, scaling, and JAX games |
| `docs/guide/deep-learning/baselines/` | Individual DRL baseline guides |
| `docs/research.md` | Paper, citation, license, and project links |
| `docs/.vitepress/config.mjs` | Base path, navigation, search, highlighting, edit links |
| `docs/.vitepress/theme/` | Styling and locally bundled Mermaid rendering |
| `docs/public/` | Public assets, including the project logo `logo.png` |
| `.github/workflows/docs.yml` | Build, verification, and Pages deployment |

## Write and verify a page

Describe the current behavior, interfaces, defaults, and usage in every page and
README. Omit comparisons with earlier versions, change narratives, and approval
history. Keep experimental results, default-selection comparisons, and provenance
in dedicated artifacts. Put feature and configuration details in `docs/`; keep
READMEs as concise overviews with links to the relevant guides.

Spell out specialized abbreviations at first use in each standalone page or
README, for example Deep Reinforcement Learning (DRL) and Proximal Policy
Optimization (PPO). Use abbreviations for later mentions and concise navigation
labels. Keep code identifiers and product names unchanged.

Use relative Markdown links between pages, for example `[Environments](./guide/environments.md)`. VitePress rewrites them for the configured base. Keep `cleanUrls: false`: explicit `.html` output works on GitHub Pages without rewrite rules.

Use site-root asset references such as `/logo.png` in Markdown and theme configuration; VitePress applies the base there. Raw `head` entries need the full `/LiteEFG/` prefix. The project logo is maintained in `docs/public/logo.png`; the repository README references the same file.

For diagrams, use a `mermaid` fenced code block with an `accTitle` and `accDescr`. Prefer small flowcharts with a nearby prose explanation.

Check Python signatures against [`LiteEFG/src/main.cpp`](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/src/main.cpp), then inspect the implementation for semantics. Account for Python wrappers that override native signatures. For baseline parameters and command-line interface (CLI) behavior, inspect the corresponding `.py` file and `baselines/utils.py`.

Keep examples small and reuse checked-in algorithms. Describe limitations where they affect use. Do not add runtime imports to the documentation build or document unbound C++ classes as callable Python interfaces.

## Automatic deployment

The [documentation workflow](https://github.com/liumy2010/LiteEFG/blob/main/.github/workflows/docs.yml) builds and checks the site for pull requests and pushes to `main`. Pull requests get validation only. Successful `main` builds upload `docs/.vitepress/dist` as a Pages artifact and deploy it to [liumy2010.github.io/LiteEFG/](https://liumy2010.github.io/LiteEFG/). You can also run the workflow manually on `main`.

One-time repository setup:

1. Open **Settings → Pages → Build and deployment**.
2. Set **Source** to **GitHub Actions**.
3. Ensure Actions can run and that any `github-pages` environment rules allow deployment from `main`.

Deployment uses the built-in `GITHUB_TOKEN` with `pages: write` and `id-token: write` in the deployment job. The build job has read-only repository access, and no personal access token is needed. Deployments are serialized without canceling a deployment in progress.

## Test selection

The Reference output checks workflow selects complete pytest modules from the changed
paths on pushes and pull requests. Pushes compare the event's before and after
commits, including every commit in that push. Pull requests compare the head
against its merge base with the target branch. Changes from multiple groups run
the union of their tests.

| Changed area | Test coverage |
| --- | --- |
| Documentation and READMEs | All executable documentation examples and coverage checks |
| Native Python baselines or saved baseline references | Native correctness, reference output, checkpoint, parallel and documentation tests |
| DarkChess, Goofspiel or PPO implementation | Tests referencing that game or algorithm, their dependent test helpers, documentation and DRL integration checks |
| Shared DRL code or models | All DRL tests, documentation and public application programming interface (API) integration checks |
| An individual test module | That module and tests that reference it, transitively |
| Shared engine or package code, dependencies, CI configuration, or an unreviewed path | Full test suite |

The routing rules live in `tests/select_ci_tests.py`. Game and algorithm selection
uses conservative source-name matching; test-helper dependencies include names
inside subprocess scripts. This is a reviewed routing policy, not a general
Python dependency analyzer. Update the rules when introducing indirect access
that the existing groups do not cover. Every selected run includes the selector's
own unit tests. Selected modules retain all parameter combinations and
multi-device checks.

Missing history, a new branch without a comparison commit, a missing selected
module, or an unreadable event selects the full suite. Changes to the selector
also require the full suite. Choose **Run workflow** on **Reference output checks**
to request a full run regardless of the diff.

The selection step prints changed paths, reasons and selected modules, and adds
them to the job summary. The test artifact includes
`artifacts/ci-selection/plan.json` and a JUnit report alongside any baseline
comparison outputs. Pytest prints each test name and the twenty slowest durations
when it finishes. Selection never records or replaces baseline references.

To preview a selection locally, use
`python tests/select_ci_tests.py --changed-files LiteEFG/baselines/drl/PPO.py`.
This writes a plan without running tests. Execute a saved plan with
`python tests/select_ci_tests.py --run-plan artifacts/ci-selection/plan.json`
after installing both test requirements files and building the native extension.
The separate documentation workflow builds and checks the website on pull
requests and pushes to `main`.

## Review checklist

- Run the production build and link check after content or configuration changes.
- Open the production preview at `/LiteEFG/`, including a deep link and the 404 page.
- Check desktop/mobile navigation, search, code blocks, diagrams, and both color themes.
- Review source changes for documentation drift, especially bindings and baseline defaults.
- Verify the Pages deployment and deep-page refreshes after deploying to GitHub.
