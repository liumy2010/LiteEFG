#include "Data/Vector.h"
#include "Data/Tensor.h"

#include "Computation/Graph.h"
#include "Computation/Checkpoint.h"
#include "Environment/Checkpoint.h"

#include "Computation/Operations.h"
#include "Computation/Static.h"
#include "Computation/Projection.h"
#include "Computation/GraphNode.h"

#include "Environment/Environment.h"
#include "Environment/NFG/NFG.h"
#include "Environment/Leduc/Leduc.h"
#include "Environment/FileEnvironment/FileEnvironment.h"

#include "Basic/BasicFunction.h"
#include "Basic/Parallel.h"

#include <iostream>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;
void BindCppEnvironment(py::module_& m);

namespace {
void BindGraphAttributes(Graph& graph, py::handle value,
                         std::unordered_set<PyObject*>& visited) {
    if(!visited.insert(value.ptr()).second) return;
    if(py::isinstance<Graph>(value) || py::isinstance<Environment>(value)) return;
    if(py::isinstance<GraphNode>(value)) {
        value.cast<GraphNode&>().owner = graph.builder;
    } else if(py::isinstance<py::dict>(value)) {
        for(auto entry : value.cast<py::dict>()) {
            BindGraphAttributes(graph, entry.first, visited);
            BindGraphAttributes(graph, entry.second, visited);
        }
    } else if(py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value) ||
              py::isinstance<py::set>(value)) {
        for(auto item : value.cast<py::iterable>()) BindGraphAttributes(graph, item, visited);
    } else {
        // Inspect stored instance data without evaluating user properties or
        // following references into module/function globals and class objects.
        if(PyType_Check(value.ptr()) || PyModule_Check(value.ptr()) ||
           PyFunction_Check(value.ptr()) || PyCFunction_Check(value.ptr()) ||
           PyMethod_Check(value.ptr())) return;
        PyObject* dictionary = PyObject_GenericGetDict(value.ptr(), nullptr);
        if(dictionary) {
            auto attributes = py::reinterpret_steal<py::object>(dictionary);
            BindGraphAttributes(graph, attributes, visited);
        } else if(PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        } else {
            throw py::error_already_set();
        }
        const auto type = py::type::of(value);
        for(auto base : type.attr("__mro__").cast<py::tuple>()) {
            for(auto entry : py::dict(base.attr("__dict__"))) {
                const auto descriptor = entry.second;
                if(Py_TYPE(descriptor.ptr()) != &PyMemberDescr_Type) continue;
                PyObject* item = Py_TYPE(descriptor.ptr())->tp_descr_get(
                    descriptor.ptr(), value.ptr(), type.ptr());
                if(item) {
                    auto member = py::reinterpret_steal<py::object>(item);
                    BindGraphAttributes(graph, member, visited);
                } else if(PyErr_ExceptionMatches(PyExc_AttributeError)) {
                    // Unassigned slots need no rebinding.
                    PyErr_Clear();
                } else {
                    throw py::error_already_set();
                }
            }
        }
    }
}

void BindGraphAttributes(Graph& graph, py::handle attributes) {
    std::unordered_set<PyObject*> visited;
    BindGraphAttributes(graph, attributes, visited);
}
}

PYBIND11_MODULE(_LiteEFG, m) {
    py::class_<GraphBuilderContext>(m, "_GraphBuilderContext");
    m.def("_graph_capture_context", &GraphNode::CaptureContext);
    m.def("_graph_restore_context", &GraphNode::RestoreContext);
    m.def("_graph_clear_context", &GraphNode::ClearContext);
    m.def("_graph_activate", &Graph::Activate);
    m.def("_graph_bind_attributes", py::overload_cast<Graph&, py::handle>(&BindGraphAttributes));
    py::class_<Vector>(m, "Vector")
        .def(py::init<>()) 
        .def("print", &Vector::Print)
        .def("__repr__",
            [](const Vector& a) {
                return a.VectorToString();
            }
        );
    //py::class_<Tensor>(m, "tensor")
    //    .def(py::init<>());
    py::class_<GraphNode>(m, "GraphNode")
        .def(py::init<>())
        .def(py::pickle(&Checkpoint::SaveNode, &Checkpoint::LoadNode))
        .def("inplace", &GraphNode::Inplace)
        .def("__add__", [](const GraphNode& a, const GraphNode::Object& b) { return a + b; })
        .def("__sub__", [](const GraphNode& a, const GraphNode::Object& b) { return a - b; })
        .def("__mul__", [](const GraphNode& a, const GraphNode::Object& b) { return a * b; })
        .def("__truediv__", [](const GraphNode& a, const GraphNode::Object& b) { return a / b; })
        .def("__pow__", [](const GraphNode& a, const GraphNode::Object& b) { return GraphNode::Pow(a, b); })
        .def("__rpow__", [](const GraphNode& exponent, const ObjectDoubleInt& base) { return GraphNode::Pow(base, exponent); })
        .def(ObjectDoubleInt() + py::self)
        .def(ObjectDoubleInt() - py::self)
        .def(ObjectDoubleInt() * py::self)
        .def(ObjectDoubleInt() / py::self)
        .def(- py::self)
        .def("__gt__", [](const GraphNode& a, const GraphNode::Object& b) { return a > b; })
        .def("__lt__", [](const GraphNode& a, const GraphNode::Object& b) { return a < b; })
        .def("__le__", [](const GraphNode& a, const GraphNode::Object& b) { return a <= b; })
        .def("__ge__", [](const GraphNode& a, const GraphNode::Object& b) { return a >= b; })
        .def("__eq__", [](const GraphNode& a, const GraphNode::Object& b) { return a == b; })
        .def(ObjectDoubleInt() < py::self)
        .def(ObjectDoubleInt() > py::self)
        .def(ObjectDoubleInt() <= py::self)
        .def(ObjectDoubleInt() >= py::self)
        .def(ObjectDoubleInt() == py::self)
        .def("sum", py::overload_cast<>(&GraphNode::Sum))
        .def("mean", py::overload_cast<>(&GraphNode::Mean))
        .def("max", py::overload_cast<>(&GraphNode::Max))
        .def("min", py::overload_cast<>(&GraphNode::Min))
        .def("copy", py::overload_cast<>(&GraphNode::Copy))
        .def("exp", py::overload_cast<>(&GraphNode::Exp))
        .def("log", py::overload_cast<>(&GraphNode::Log))
        .def("argmax", py::overload_cast<>(&GraphNode::Argmax))
        .def("argmin", py::overload_cast<>(&GraphNode::Argmin))
        .def("euclidean", py::overload_cast<>(&GraphNode::Euclidean))
        .def("negative_entropy", py::overload_cast<const bool&>(&GraphNode::NegativeEntropy), py::arg("shifted")=false)
        .def("normalize", py::overload_cast<const double&, const bool&>(&GraphNode::Normalize), py::arg("p_norm"), py::arg("ignore_negative") = false)
        .def("dot", py::overload_cast<const GraphNode&>(&GraphNode::Dot))
        .def("project", py::overload_cast<const std::string&, const GraphNode::Object&>(&GraphNode::Project), py::arg("distance"), py::arg("gamma")=0.0)
        .def("project", py::overload_cast<const std::string&, const GraphNode::Object&, const GraphNode&>(&GraphNode::Project), py::arg("distance"), py::arg("gamma"), py::arg("mu"));
    
    m.def("const", GraphNode::ConstVector, py::arg("size"), py::arg("val"));
    m.def("sum", py::overload_cast<const GraphNode&>(GraphNode::Sum));
    m.def("mean", py::overload_cast<const GraphNode&>(GraphNode::Mean));
    m.def("max", py::overload_cast<const GraphNode&>(GraphNode::Max));
    m.def("min", py::overload_cast<const GraphNode&>(GraphNode::Min));
    m.def("copy", py::overload_cast<const GraphNode&>(GraphNode::Copy));
    m.def("exp", py::overload_cast<const GraphNode&>(GraphNode::Exp));
    m.def("log", py::overload_cast<const GraphNode&>(GraphNode::Log));
    m.def("argmax", py::overload_cast<const GraphNode&>(GraphNode::Argmax));
    m.def("argmin", py::overload_cast<const GraphNode&>(GraphNode::Argmin));
    m.def("euclidean", py::overload_cast<const GraphNode&>(GraphNode::Euclidean));
    m.def("negative_entropy", py::overload_cast<const GraphNode&, const bool&>(GraphNode::NegativeEntropy), py::arg(), py::arg("shifted")=false);
    m.def("normalize", py::overload_cast<const GraphNode&, const double&, const bool&>(GraphNode::Normalize), py::arg(), py::arg("p_norm"), py::arg("ignore_negative") = false);
    m.def("dot", py::overload_cast<const GraphNode&, const GraphNode&>(GraphNode::Dot));
    m.def("maximum", py::overload_cast<const GraphNode&, const GraphNode::Object&>(GraphNode::Maximum));
    m.def("minimum", py::overload_cast<const GraphNode&, const GraphNode::Object&>(GraphNode::Minimum));
    m.def("aggregate", py::overload_cast<const GraphNode&, const std::string&, const std::string&, const std::string&, const double&>(GraphNode::Aggregate), py::arg(), py::arg("aggregator"), py::arg("object")="children", py::arg("player")="self", py::arg("padding")=0.0);
    m.def("project", py::overload_cast<const GraphNode&, const std::string&, const GraphNode::Object&>(GraphNode::Project), py::arg(), py::arg("distance"), py::arg("gamma")=0.0);
    m.def("project", py::overload_cast<const GraphNode&, const std::string&, const GraphNode::Object&, const GraphNode&>(GraphNode::Project), py::arg(), py::arg("distance"), py::arg("gamma"), py::arg("mu"));
    m.def("cat", py::overload_cast<const std::vector<GraphNode>&>(GraphNode::Concat), py::arg("nodes"));

    m.def("set_seed", Basic::SetSeed, py::arg("seed"));
    m.def("set_threads", Parallel::SetThreads, py::arg("threads"),
          "Set the process-wide number of native computation threads (default: 1).");
    m.def("get_threads", Parallel::GetThreads,
          "Return the configured number of native computation threads.");
    m.def("_uniform", GraphNode::RandomUniform, py::arg("node"), py::arg("lower") = 0.0, py::arg("upper") = 1.0);
    m.def("_normal", GraphNode::RandomNormal, py::arg("node"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);
    m.def("_exponential", GraphNode::RandomExponential, py::arg("node"), py::arg("lambda_") = 1.0);
    

    py::class_<Graph>(m, "Graph", py::dynamic_attr())
        .def(py::init<>())
        .def(py::pickle(
            [](const py::object& self) {
                auto attributes = py::dict(self.attr("__dict__").attr("copy")());
                attributes.attr("pop")("_checkpoint_environment", py::none());
                return py::make_tuple(Checkpoint::SaveGraph(self.cast<const Graph&>()), attributes);
            },
            [](const py::tuple& state) {
                if (state.size() != 2) throw std::invalid_argument("Invalid checkpoint graph state");
                auto graph = Checkpoint::LoadGraph(state[0].cast<py::dict>());
                auto attributes = state[1].cast<py::dict>();
                BindGraphAttributes(graph, attributes);
                return std::make_pair(std::move(graph), attributes);
            }))
        .def_readonly("utility", &Graph::utility)
        .def_readonly("opponent_reach_prob", &Graph::opponent_reach_prob)
        .def_readonly("reach_prob", &Graph::reach_prob)
        .def_readonly("action_set_size", &Graph::action_set_size)
        .def_readonly("subtree_size", &Graph::subtree_size);

    py::class_<GraphNodeStatus, std::shared_ptr<GraphNodeStatus>>(m, "GraphNodeStatus")
        .def(py::init<>())
        .def("__enter__", &GraphNodeStatus::Enter, py::return_value_policy::reference)
        .def("__exit__", &GraphNodeStatus::Exit);

    py::class_<ForwardNodeStatus, GraphNodeStatus, std::shared_ptr<ForwardNodeStatus>>(m, "forward")
        .def(py::init<const bool&, const int&>(), py::arg("is_static") = false, py::arg("color") = 0);
    
    py::class_<BackwardNodeStatus, GraphNodeStatus, std::shared_ptr<BackwardNodeStatus>>(m, "backward")
        .def(py::init<const bool&, const int&>(), py::arg("is_static") = false, py::arg("color") = 0);

    py::class_<Environment, std::shared_ptr<Environment>>(m, "Environment")
        .def("set_graph", [](py::object self, py::object graph) {
            self.cast<Environment&>().SetGraph(graph.cast<const Graph&>());
            graph.attr("_checkpoint_environment") = self;
        }, py::arg("graph"))
        .def("update", py::overload_cast<const GraphNode&, const int&, std::vector<int>, const std::string&>(&Environment::Update), py::arg("strategy"), py::arg("upd_player") = -1, py::arg("upd_color")=std::vector<int>{-1}, py::arg("traverse_type")="default")
        .def("update", py::overload_cast<std::vector<GraphNode>, const int&, std::vector<int>, const std::string&>(&Environment::Update), py::arg("strategies"), py::arg("upd_player") = -1, py::arg("upd_color")=std::vector<int>{-1}, py::arg("traverse_type")="default")
        .def("update_strategy", py::overload_cast<const GraphNode&, const bool&>(&Environment::UpdateStrategy), py::arg("strategy"), py::arg("update_best") = false)
        .def("exploitability", py::overload_cast<const GraphNode&, const std::string&>(&Environment::Exploitability), py::arg("strategy"), py::arg("type_name") = "default")
        .def("exploitability", py::overload_cast<const std::vector<GraphNode>&, const std::string&>(&Environment::Exploitability), py::arg("strategy"), py::arg("type_name") = "default")
        .def("utility", py::overload_cast<const GraphNode&, const std::string&>(&Environment::Utility), py::arg("strategy"), py::arg("type_name") = "default")
        .def("utility", py::overload_cast<const std::vector<GraphNode>&, const std::string&>(&Environment::Utility), py::arg("strategy"), py::arg("type_name") = "default")
        .def("get_value", &Environment::GetValue, py::arg("player"), py::arg("node"))
        .def("get_strategy", &Environment::GetStrategy, py::arg("player"), py::arg("strategy"), py::arg("type_name") = "default")
        .def("set_value", py::overload_cast<const int&, const GraphNode&, const std::vector<std::vector<double>>&>(&Environment::SetValue), py::arg("player"), py::arg("node"), py::arg("values"))
        .def("set_value", py::overload_cast<const int&, const GraphNode&, const std::vector<double>&>(&Environment::SetValue), py::arg("player"), py::arg("node"), py::arg("values"));

    py::class_<FileEnvironment, Environment, std::shared_ptr<FileEnvironment>>(m, "FileEnv")
        .def(py::init<const std::string&, const std::string&>(), py::arg("file_name"), py::arg("traverse_type") = "Enumerate");

    m.def("_checkpoint_save_environment", &Checkpoint::SaveEnvironment);
    m.def("_checkpoint_load_environment", &Checkpoint::LoadEnvironment);
    m.def("_checkpoint_restore", &Checkpoint::Restore);
    m.def("_checkpoint_get_random_state", &Checkpoint::SaveRandomState);
    m.def("_checkpoint_set_random_state", &Checkpoint::LoadRandomState);
    m.def("_checkpoint_activate_graph", &Graph::Activate);
    BindCppEnvironment(m);
}
