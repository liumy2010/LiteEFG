#ifndef CHECKPOINT_H_
#define CHECKPOINT_H_

#include "Computation/Graph.h"

#include <pybind11/pybind11.h>

// Plain Python containers form the versioned, pickleable native state. Loading
// never executes graph operations or initializes an environment.
namespace Checkpoint {

pybind11::dict SaveGraph(const Graph& graph);
Graph LoadGraph(const pybind11::dict& state);
pybind11::tuple SaveNode(const GraphNode& node);
GraphNode LoadNode(const pybind11::tuple& state);
pybind11::dict SaveRandomState();
void LoadRandomState(const pybind11::dict& state);

}

#endif
