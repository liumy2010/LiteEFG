#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "Data/Vector.h"

#include <string>
#include <memory>

class Operation {
public:
    bool is_static = false;
    std::string name;
    Vector tmp, info; // tmp is an auxiliary vector, info (optional) stores the information of the operation

    Operation(const std::string& name_, const bool& is_static_=false) : name{name_}, is_static{is_static_} {}
    virtual void Execute(Vector& result, const std::vector<Vector*>& inputs) = 0;
    virtual ~Operation() {}
};

class CopyOperation : public Operation {
public:
    CopyOperation(const bool& is_static_=false) : Operation("Copy", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class AddOperation : public Operation {
public:
    AddOperation(const bool& is_static_=false) : Operation("Add", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class SubOperation : public Operation {
public:
    SubOperation(const bool& is_static_=false) : Operation("Sub", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MulOperation : public Operation {
public:
    MulOperation(const bool& is_static_=false) : Operation("Mul", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class DivOperation : public Operation {
public:
    DivOperation(const bool& is_static_=false) : Operation("Div", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class ExpOperation : public Operation {
public:
    ExpOperation(const bool& is_static_=false) : Operation("Exp", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class LogOperation : public Operation {
public:
    LogOperation(const bool& is_static_=false) : Operation("Log", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class SumOperation : public Operation {
public:
    SumOperation(const bool& is_static_=false) : Operation("Sum", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MeanOperation : public Operation {
public:
    MeanOperation(const bool& is_static_=false) : Operation("Mean", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MaxOperation : public Operation {
public:
    MaxOperation(const bool& is_static_=false) : Operation("Max", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MinOperation : public Operation {
public:
    MinOperation(const bool& is_static_=false) : Operation("Min", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class DotOperation : public Operation {
public:
    DotOperation(const bool& is_static_=false) : Operation("Dot", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class ArgmaxOperation : public Operation {
public:
    ArgmaxOperation(const bool& is_static_=false) : Operation("Argmax", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class ArgminOperation : public Operation {
public:
    ArgminOperation(const bool& is_static_=false) : Operation("Argmin", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MaximumOperation : public Operation {
public:
    MaximumOperation(const bool& is_static_=false) : Operation("Maximum", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class MinimumOperation : public Operation {
public:
    MinimumOperation(const bool& is_static_=false) : Operation("Minimum", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class EuclideanOperation : public Operation {
public:
    EuclideanOperation(const bool& is_static_=false) : Operation("Euclidean", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class NegativeEntropyOperation : public Operation {
public:
    bool shifted;
    NegativeEntropyOperation(const bool& shifted_=false, const bool& is_static_=false) : Operation("NegativeEntropy", is_static_), shifted{shifted_} {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class AggregateOperation : public Operation {
public:
    enum InfoIndex {
        padding = 0,
        object = 1,
        player = 2,
    };
    std::shared_ptr<Operation> aggregator;
    AggregateOperation(std::shared_ptr<Operation> aggregator_, const std::string& object="children", const std::string& player="self", const double& padding=0.0, const bool& is_static_=false);
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
}; 

class NormalizeOperation : public Operation {
public:
    double p_norm;
    bool ignore_negative;
    NormalizeOperation(const double& p_norm_, const bool& ignore_negative_=false, const bool& is_static_=false);
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class CompareOperation : public Operation {
int type;
public:
    enum ComparatorType {
        start = 0,
        greater_than = 0,
        greater_than_or_equal = 1,
        less_than = 2,
        less_than_or_equal = 3,
        equal = 4,
        type_num = 5,
    };
    CompareOperation(const int& type_, const bool& is_static_=false);
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

class PowOperation : public Operation {
public:
    PowOperation(const bool& is_static_=false) : Operation("Pow", is_static_) {}
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

#endif
