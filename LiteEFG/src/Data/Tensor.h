#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>
#include <initializer_list>

class Tensor{
public:
    int dim;
    std::vector<int> shape;
    std::vector<double> data;

    Tensor() : dim{0} {}
    Tensor(const std::vector<double> data_, const std::vector<int>& shape_);
    double operator[](const std::vector<int>& idx);
};

#endif