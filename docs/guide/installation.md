---
description: Install LiteEFG with tabular and deep reinforcement learning support, build from source, and configure JAX for processors or NVIDIA graphics cards.
---

# Installation

LiteEFG is a Python package with a C++17 extension. The default installation includes tabular solvers and Deep Reinforcement Learning (DRL) support.

## Install the package

Use a virtual environment with Python 3.10 or newer. Availability of a prebuilt wheel depends on your platform and Python version.

::: code-group

```sh [macOS / Linux]
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install LiteEFG
```

```powershell [Windows PowerShell]
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install LiteEFG
```

:::

If pip builds LiteEFG from source, you also need the native tools described below.

### Deep learning with JAX

The default installation includes JAX, Optax, and Flax for the
[deep learning examples](./deep-learning.md). Use Python 3.10 or newer and install
from PyPI:

```sh
python -m pip install LiteEFG
```

For an NVIDIA graphics processing unit (GPU), use Linux or Ubuntu under Windows Subsystem for Linux 2 (WSL2) and install an appropriate
NVIDIA driver. Select the CUDA backend through JAX. Choose one of the following
commands.

For CUDA 12:

```sh
python -m pip install LiteEFG 'jax[cuda12]>=0.6.2,<0.8'
```

For CUDA 13, use Python 3.11 or newer and JAX 0.7.2:

```sh
python -m pip install LiteEFG 'jax[cuda13]>=0.7.2,<0.8'
```

JAX's CUDA extras install the matching CUDA runtime libraries through pip; a
separate system CUDA toolkit is not required. The NVIDIA driver and GPU must
support the chosen backend. See [JAX installation](https://docs.jax.dev/en/latest/installation.html)
for driver and GPU requirements. Installing LiteEFG alone does not install a CUDA
backend, but it can use a compatible JAX GPU backend already in the environment.

Run GPU installation commands inside WSL2 on Windows. JAX selects an available
device; inspect `jax.devices()` to confirm the selection. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX to disable its default
GPU memory preallocation.

## Build the current repository

Clone with submodules to include the required `pybind11/` directory:

```sh
git clone --recursive https://github.com/liumy2010/LiteEFG.git
cd LiteEFG
python -m pip install .
```

For an existing clone with an empty `pybind11/` directory:

```sh
git submodule update --init --recursive
python -m pip install .
```

The native build needs a compiler with C++17 support: for example, Xcode Command Line Tools on macOS, GCC/Clang on Linux, or the Visual Studio C++ build tools on Windows.

Pip installs the Python build tools automatically.

### Deep learning from the repository

For central processing unit (CPU) execution, use the standard source installation:

```sh
python -m pip install .
```

For CUDA 12:

```sh
python -m pip install . 'jax[cuda12]>=0.6.2,<0.8'
```

For CUDA 13, use Python 3.11 or newer:

```sh
python -m pip install . 'jax[cuda13]>=0.7.2,<0.8'
```

The [JAX backend requirements](#deep-learning-with-jax), driver requirements,
and device configuration above also apply to source installations.

## Check the installation

Run the import check outside the repository directory when using `pip install .`. Otherwise, Python may import the local source package, which may not contain the installed extension.

```sh
python -c "import LiteEFG as leg; import pyspiel; print(leg.__file__); print(pyspiel.load_game('kuhn_poker'))"
```

Continue with the [Counterfactual Regret Minimization (CFR) quick start](./quick-start.md) for tabular solvers or
[Deep learning with JAX](./deep-learning.md) for neural policies.

## Run the baseline scripts

The command-line examples require an additional dependency:

```sh
python -m pip install tqdm
```

See [examples](./examples.md) for the command-line runners.

## Common installation issues

| Symptom | Check |
| --- | --- |
| CMake cannot find `pybind11` | Initialize the Git submodule, then rerun the install. |
| Compiler or build-generator error | Check that a compiler with C++17 support is installed and available in your shell. Pip provides build dependencies, not a system compiler. |
| `No module named 'LiteEFG._LiteEFG'` | Check the active Python environment and whether the current directory is shadowing an installed package. For development, an editable install (`python -m pip install -e .`) can build the extension for the checkout. |
| Missing `pyspiel` or `pandas` | Install with dependencies enabled in the same environment. |
| Random bindings are unavailable | Rebuild the current checkout. |
| First OpenSpiel environment creation takes time | The adapter enumerates and caches the game. Start with Kuhn poker; see [conversion and caching](./environments.md). |

## Build only the documentation

Follow [contributing to the docs](../contributing.md) to build the website with Node.js.
