#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <initializer_list>
#include <string>

class Vector {
private:
    std::vector<double> elements;

public:
    int size;

    Vector() : size{0} {}
    Vector(const std::initializer_list<double>& init);
    Vector(const std::vector<double>& init);
    Vector(const int& n, const double& val);
    Vector(const Vector& rhs);

    void operator=(const Vector& rhs);
    bool operator==(const Vector& rhs) const;

    void Add(const Vector& rhs);
    void Sub(const Vector& rhs);
    void Mul(const double& scalar);
    void Mul(const Vector& scalar);
    void Div(const double& scalar);
    void Div(const Vector& scalar);

    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector operator*(const double& scalar) const;
    Vector operator*(const Vector& scalar) const;
    Vector operator/(const double& scalar) const;
    Vector operator/(const Vector& scalar) const;

    double& operator[](const int& index);
    double operator[](const int& index) const;

    friend Vector operator*(const double& scalar, const Vector& rhs);

    friend Vector operator/(const double& scalar, const Vector& rhs);

    double Dot(const Vector& rhs) const;
    double Sum() const;
    void Concat(const double& rhs);
    void Concat(const Vector& rhs);
    void push_back(const double& val);
    void Resize(const int& n, const double& val=0.0);
    void Set(const double& val);

    void Print() const;
    std::string VectorToString() const;
};

#endif