#ifndef STATIC_H_
#define STATIC_H_

#include "Data/Vector.h"
#include "Operations.h"

#include <string>
#include <memory>
#include <variant>

using Variant = std::variant<int, double, std::string>;

class Static : public Operation {
public:
    Vector elements;

    Static(const std::string& name_) : Operation(name_, true) {}

    virtual void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
    virtual ~Static() {}
};

class StaticConstVector : public Static {
public:
    StaticConstVector() : Static("StaticConstVector"){}
    StaticConstVector(const int& size, const double& val);
    StaticConstVector(const std::vector<Variant>& inputs);
    StaticConstVector(const Vector& inputs);
    
    bool operator==(const StaticConstVector& rhs) const {
        return elements == rhs.elements;
    }
    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

#endif