#ifndef ENVIRONMENT_CHECKPOINT_H
#define ENVIRONMENT_CHECKPOINT_H

#include "Environment/Environment.h"
#include <pybind11/pybind11.h>

namespace Checkpoint {
pybind11::dict SaveEnvironment(const Environment& env, const Graph& graph);
std::shared_ptr<Environment> LoadEnvironment(const pybind11::dict& state);
void Restore(Environment& env, Graph& graph,
             const Environment& saved_env, const Graph& saved_graph);
}

#endif
