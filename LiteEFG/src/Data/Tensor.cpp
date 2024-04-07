#include "Data/Tensor.h"

Tensor::Tensor(const std::vector<double> data_, const std::vector<int>& shape_) : shape(shape_), data(data_){
    dim = shape_.size();
}

double Tensor::operator[](const std::vector<int>& idx){
    int index = 0;
    for(int i=0; i<dim; ++i){
        index = index * shape[i] + idx[i];
    }
    return data[index];
}