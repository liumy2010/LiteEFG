#include "Static.h"

#include <cmath>
#include <stdexcept>

void Static::Execute(Vector& result, const std::vector<Vector*>& inputs){
    result = elements;
}

StaticConstVector::StaticConstVector(const int& size, const double& val) : Static("StaticConstVector") {
    elements = Vector(size, val);
}

StaticConstVector::StaticConstVector(const std::vector<Variant>& inputs) : Static("StaticConstVector") {
    for (auto& elem : inputs) {
        if (! std::holds_alternative<double>(elem)){
            throw std::invalid_argument("The arguments of CustomizedVector must be floats");
        }
    }

    std::vector<double> input_values;
    for (auto& elem : inputs) {
        input_values.push_back(std::get<double>(elem));
    }
    elements = Vector(input_values);
}

StaticConstVector::StaticConstVector(const Vector& inputs) : Static("StaticConstVector") {
    elements = inputs;
}

void StaticConstVector::Execute(Vector& result, const std::vector<Vector*>& inputs){
    if(inputs.size() != 2){
        if(inputs.size() == 0) {result = elements; return;}
        throw std::invalid_argument("ConstVector only allows either no input or two inputs (size, val)");
    }
    if(inputs[0] -> size != 1 || inputs[1] -> size != 1){
        throw std::invalid_argument("The inputs of ConstVector must be scalars");
    }
    result.Resize(round((*inputs[0])[0]), (*inputs[1])[0]);
}