#include "Data/Vector.h"
#include "Data/Tensor.h"

#include "Computation/Graph.h"

#include "Computation/Operations.h"
#include "Computation/Static.h"
#include "Computation/Projection.h"
#include "Computation/GraphNode.h"

#include "Environment/Environment.h"
#include "Environment/NFG/NFG.h"
#include "Environment/Leduc/Leduc.h"
#include "Environment/FileEnvironment/FileEnvironment.h"

#include "Basic/BasicFunction.h"

#include <iostream>
/*
int main(){
    std::string file_name;
    std::cin>>file_name;
    static Environment* env = new FileEnvironment("../GameInstances/"+file_name, "Enumerate");

    auto graph = Graph();
    
    auto eta = 0.1;
    auto gamma = 0.0;
    auto tau = 0.0;

    BackwardNodeStatus backward_node_static(true, 2);
    backward_node_static.Enter();

    auto alpha = 1.0;

    auto ev = GraphNode::ConstVector(1, 0.0);
    auto coef = alpha * tau;
    auto mu = graph.subtree_size.Normalize(1.0, true);
    auto eta_coef = eta * coef;
    
    auto u = GraphNode::ConstVector(graph.action_set_size, 1.0 / graph.action_set_size);
    auto bar_u = u.Copy();

    BackwardNodeStatus backward_node(true, 2);
    backward_node.Enter();

    auto m_th = graph.opponent_reach_prob;
    
    auto eta_tau = eta_coef / m_th + 1;
    auto gradient = GraphNode::Aggregate(ev, "sum", "children", "opponents") + graph.utility;
    ev.Inplace(GraphNode::Dot(gradient, u));

    GraphNode::Aggregate(u, "sum", "parent", "opponents");

    ev.Inplace(ev - GraphNode::Euclidean(u) * coef);

    gradient.Inplace(gradient / m_th * eta);

    u.Inplace((bar_u + gradient) / eta_tau);
    u.Inplace(u.Project("L2", gamma, mu));

    env->SetGraph(graph);

    for(int i=0;i<1000;i++){
        env -> UpdateStrategy(u, true);
        env -> Update(u, -1);
        if(i % 1000 == 0){
            auto exploitability = env -> Exploitability(u, "best-iterate");
            auto utility = env -> Exploitability(u, "avg-iterate");
            printf("%d %.12lf %.12lf %.12lf %.12lf\n", i, exploitability[0], exploitability[1], utility[0], utility[1]);
        }
    }
    env->GetValue(1, u);

    return 0;
}
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
//#include <pybind11/memory.h>

namespace py = pybind11;

PYBIND11_MODULE(_LiteEFG, m) {
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
        .def("inplace", &GraphNode::Inplace)
        .def("__add__", [](const GraphNode& a, const GraphNode::Object& b) { return a + b; })
        .def("__sub__", [](const GraphNode& a, const GraphNode::Object& b) { return a - b; })
        .def("__mul__", [](const GraphNode& a, const GraphNode::Object& b) { return a * b; })
        .def("__truediv__", [](const GraphNode& a, const GraphNode::Object& b) { return a / b; })
        .def("__pow__", [](const GraphNode& a, const GraphNode::Object& b) { return GraphNode::Pow(a, b); })
        .def("__rpow__", [](const ObjectDoubleInt& a, const GraphNode& b) { return GraphNode::Pow(a, b); })
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

    m.def("set_seed", Basic::SetSeed, py::arg("seed"));

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
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
        .def("set_graph", &Environment::SetGraph, py::arg("graph"))
        .def("update", py::overload_cast<const GraphNode&, const int&, std::vector<int>>(&Environment::Update), py::arg("strategy"), py::arg("upd_player") = -1, py::arg("upd_color")=std::vector<int>{-1})
        .def("update", py::overload_cast<std::vector<GraphNode>, const int&, std::vector<int>>(&Environment::Update), py::arg("strategies"), py::arg("upd_player") = -1, py::arg("upd_color")=std::vector<int>{-1})
        .def("update_strategy", py::overload_cast<const GraphNode&, const bool&>(&Environment::UpdateStrategy), py::arg("strategy"), py::arg("update_best") = false)
        .def("exploitability", py::overload_cast<const GraphNode&, const std::string&>(&Environment::Exploitability), py::arg("strategy"), py::arg("type_name") = "default")
        .def("utility", py::overload_cast<const GraphNode&, const std::string&>(&Environment::Utility), py::arg("strategy"), py::arg("type_name") = "default")
        .def("get_value", &Environment::GetValue, py::arg("player"), py::arg("node"))
        .def("get_strategy", &Environment::GetStrategy, py::arg("player"), py::arg("strategy"), py::arg("type_name") = "default");
    //py::class_<NFG, Environment, std::shared_ptr<NFG>>(m, "NFG")
    //    .def(py::init<const int&, const std::vector<Tensor>&>(), py::arg("num_players"), py::arg("utility_matrix"));
    //py::class_<Leduc, Environment, std::shared_ptr<Leduc>>(m, "Leduc")
    //    .def(py::init<const int&, const int&, const int&, const std::string&>(), py::arg("num_players") = 2, py::arg("num_cards") = 3, py::arg("num_suits") = 2, py::arg("traverse_type") = "Enumerate");
    py::class_<FileEnvironment, Environment, std::shared_ptr<FileEnvironment>>(m, "FileEnv")
        .def(py::init<const std::string&, const std::string&>(), py::arg("file_name"), py::arg("traverse_type") = "Enumerate");
}
