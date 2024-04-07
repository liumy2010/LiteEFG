#include "Vector.h"

#include "Basic/Constants.h"

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

Vector::Vector(const std::initializer_list<double>& init) : elements(init) {size = init.size();}

Vector::Vector(const std::vector<double>& init) : elements(init) {size = init.size();}

Vector::Vector(const int& n, const double& val) : size{n} {for(int i=0;i<n;i++) elements.push_back(val);}

Vector::Vector(const Vector& rhs) : size(rhs.size) {
    elements.resize(rhs.size);
    for(int i=0;i<size;++i) elements[i] = rhs[i];
}

void Vector::operator=(const Vector& rhs) {
    if(elements.size() < rhs.size){ // Resize if necessary
        elements.resize(rhs.size * 2);
    }
    size = rhs.size;
    for(int i=0;i<size;++i) elements[i] = rhs[i];
}

bool Vector::operator==(const Vector& rhs) const {
    if(size != rhs.size) return false;
    for(int i=0;i<size;++i) if(elements[i] != rhs[i]) return false;
    return true;
}

void Vector::Add(const Vector& rhs) {
    /*
        Supports vector + vector, vector + scalar, and scalar + vector
    */
    if (size != rhs.size && size != 1 && rhs.size != 1) {
        throw std::invalid_argument("In addition, vectors must be of the same size or one of them must be a scalar");
    }
    if(size == 1){
        Resize(rhs.size, elements[0]);
    }
    for (size_t i = 0; i < size; ++i) {
        elements[i] += (rhs.size==1) ? rhs[0] : rhs[i];
    }
}

void Vector::Sub(const Vector& rhs) {
    /*
        Supports vector - vector, vector - scalar, and scalar - vector
    */
    if (size != rhs.size && size != 1 && rhs.size != 1) {
        throw std::invalid_argument("In subtraction, vectors must be of the same size or one of them must be a scalar");
    }
    if(size == 1){
        Resize(rhs.size, elements[0]);
    }
    for (size_t i = 0; i < size; ++i) {
        elements[i] -= (rhs.size==1) ? rhs[0] : rhs[i];
    }
}

void Vector::Mul(const double& scalar) {
    for (double& elem : elements) {
        elem *= scalar;
    }
}

void Vector::Mul(const Vector& rhs) {
    /*
        Supports vector * vector (same size), vector * scalar, and scalar * vector
    */
    if (size != 1 && rhs.size != 1 && size != rhs.size) {
        throw std::invalid_argument("In multiplication, one of them must be a scalar or they must be of the same size");
    }
    if(size == 1) {
        Resize(rhs.size, elements[0]);
    }
    for (size_t i = 0; i < size; ++i) {
        elements[i] *= (rhs.size==1) ? rhs[0] : rhs[i];
    }
}

void Vector::Div(const double& scalar) {
    /*if (std::fabs(scalar) <= Constants::EPS) {
        throw std::invalid_argument("Division by zero");
    }*/
    double div_scalar = (std::fabs(scalar) < Constants::EPS) ? Constants::EPS : scalar;
    for (double& elem : elements) {
        elem /= div_scalar;
    }
}

void Vector::Div(const Vector& rhs) {
    /*
        Supports vector / vector (same size), vector / scalar, and scalar / vector
    */
    if (size != 1 && rhs.size != 1 && size != rhs.size) {
        throw std::invalid_argument("In division, one of them must be a scalar or they must be of the same size");
    }
    if(size == 1) {
        Resize(rhs.size, elements[0]);
    }
    for (size_t i = 0; i < size; ++i) {
        double x = (rhs.size==1) ? rhs[0] : rhs[i];
        elements[i] /= (std::fabs(x) < Constants::EPS) ? (x<0.0?-Constants::EPS:Constants::EPS) : x;
    }
}

Vector Vector::operator+(const Vector& rhs) const {
    Vector result(*this);
    result.Add(rhs);
    return result;
}

Vector Vector::operator-(const Vector& rhs) const {
    Vector result(*this);
    result.Sub(rhs);
    return result;
}

Vector Vector::operator*(const double& scalar) const {
    Vector result(*this); // Copy current vector
    result.Mul(scalar);
    return result;
}

Vector Vector::operator*(const Vector& rhs) const {
    Vector result(*this); // Copy current vector
    result.Mul(rhs);
    return result;
}

Vector Vector::operator/(const double& scalar) const {
    Vector result(*this); // Copy current vector
    result.Div(scalar);
    return result;
}

Vector Vector::operator/(const Vector& rhs) const {
    Vector result(*this); // Copy current vector
    result.Div(rhs);
    return result;
}

double& Vector::operator[](const int& index) {
    if (index < 0 || index >= size) {
        throw std::invalid_argument("Index out of range");
    }
    return elements[index];
}

double Vector::operator[](const int& index) const {
    if (index < 0 || index >= size) {
        throw std::invalid_argument("Index out of range");
    }
    return elements[index];
}

double Vector::Dot(const Vector& rhs) const {
    if (size != rhs.size) {
        throw std::invalid_argument("In inner product, vectors must be of the same size");
    }
    double result = 0.0;
    for (size_t i = 0; i < size; ++i) {
        result += elements[i] * rhs[i];
    }
    return result;
}

double Vector::Sum() const {
    double result = 0.0;
    for (double elem : elements) {
        result += elem;
    }
    return result;
}

void Vector::Concat(const double& rhs) {
    Resize(size + 1);
    elements[size-1] = rhs;
}

void Vector::Concat(const Vector& rhs) {
    int cur_size = size;
    Resize(size + rhs.size);
    for(int i=0;i<rhs.size;++i) elements[i+cur_size] = rhs[i];
}

void Vector::push_back(const double& val) {
    if(elements.size() == size) elements.push_back(val);
    else elements[size] = val;
    size++;
}

void Vector::Print() const {
    std::cout<<Vector::VectorToString()<<std::endl;
}

std::string Vector::VectorToString() const {
    std::string ss="";
    for(size_t i = 0; i < size; ++i) {
        if(i != 0) ss+= ", ";
        ss += std::to_string(elements[i]);
    }
    
    ss = "(" + ss + ")";
    return ss;
}

Vector operator*(const double& scalar, const Vector& rhs) {
    return rhs * scalar;
}

Vector operator/(const double& scalar, const Vector& rhs) {
    Vector result(rhs);
    for (int i=0;i<rhs.size;++i) {
        result[i] = (std::fabs(result[i]) < Constants::EPS) ? scalar / ((result[i] < 0.0)?-Constants::EPS:Constants::EPS) : scalar / result[i];
    }
    return result;
}

void Vector::Resize(const int& n, const double& val) {
    if(n < 0) {
        throw std::invalid_argument("Invalid size");
    }
    if(n > elements.size()) {
        elements.resize(n*2, val); // double the size to decrease the number of resizes
    }
    for(int i=size;i<n;++i) elements[i] = val;
    size = n;
}

void Vector::Set(const double& val) {
    for(int i=0;i<size;++i) elements[i] = val;
}