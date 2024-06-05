#include "Basic/BasicFunction.h"

#include "Basic/Constants.h"

#include <cmath>
#include <stdexcept>

std::default_random_engine Basic::generator;
std::uniform_real_distribution<double> Basic::uniform(0.0,1.0);

void Basic::SetSeed(const unsigned int& seed){
    generator.seed(seed);
}

double Basic::Sqr(const double& x){
    return x*x;
}

int Basic::Sample(const std::vector<double>& probs){
    Vector a = Vector(probs);
    return Sample(&a);
}

int Basic::Sample(Vector* probs){
    double sum = 0.0;
    for(int i=0;i<probs->size;i++){
        sum += (*probs)[i];
    }
    if(fabs(sum - 1.0) > Constants::EPS){
        throw std::invalid_argument("Probabilities do not sum to 1");
    }
    double r = Basic::uniform(Basic::generator);
    sum = 0.0;
    for(int i=0;i<probs->size;i++){
        sum += (*probs)[i];
        if(r<sum) return i;
    }
    return probs->size-1;
}

bool Basic::IsPrefixString(const std::string& a, const std::string& prefix){
    if(a.size() < prefix.size()) return false;
    for(int i=0;i<prefix.size();i++){
        if(a[i] != prefix[i]) return false;
    }
    return true;
}

std::string Basic::GetSlice(const std::string& a, const int& start, const char& end){
    int i = start;
    while(i < a.size() && a[i] != end) i++;
    return a.substr(start, i-start);
}