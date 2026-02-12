#ifndef BASICFUNCTION_H_
#define BASICFUNCTION_H_

#include "Data/Vector.h"

#include <vector>
#include <random>
#include <string>

namespace Basic{

extern std::default_random_engine generator;
extern std::uniform_real_distribution<double> uniform;
extern std::normal_distribution<double> normal;
extern std::exponential_distribution<double> exponential;

void SetSeed(const unsigned int& seed);

double Sqr(const double& x);

int Sample(const std::vector<double>& probs);
int Sample(Vector* probs);

bool IsPrefixString(const std::string& a, const std::string& prefix);
std::string GetSlice(const std::string& a, const int& start, const char& end);

};


#endif