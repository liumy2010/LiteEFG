#include "Projection.h"

#include "Basic/Constants.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

ProjectionOperation::ProjectionOperation(const std::string &distance_name_, const bool& is_static_)
                        : Operation("Projection", is_static_), distance_name(distance_name_) {
    if(distance_name_ != "L2" && distance_name_ != "KL") {
        throw std::invalid_argument("Invalid distance name : " + distance_name_ + ". Only L2 and KL are supported");
    }
}

void SparseMax(Vector& strategy, const double& gamma, const Vector& lowerbound){
    int n = strategy.size;

    double* aux = new double[n];
    for(int i=0;i<n;i++) aux[i] = strategy[i];
    std::sort(aux, aux + n);

    double tau = 0.0, C = - aux[0] + (1.0 - gamma) / double(n), cur_sum=0.0;
    for(int i = 0; i < n; ++i)
        aux[i] += C,
        cur_sum += aux[i];

    for(int i = 0; i < n; ++i){
        double sum = - (1.0 - gamma) + cur_sum;
        if(sum < aux[i] * (n - i)) {
            tau = sum / (n - i);
            break;
        }
        cur_sum -= aux[i];
    }
    cur_sum = 0.0;
    for(int i = 0; i < n; ++i) 
        strategy[i] = std::max(strategy[i]- tau + C, 0.0) + gamma * lowerbound[i],
        cur_sum += strategy[i];
    for(int i=0; i<n; ++i) strategy[i] /= cur_sum; // normalize to avoid numerical issue
    return;
}

void EntropyMax(Vector& strategy, const double& gamma, const Vector& lowerbound){
    int n = strategy.size;
    std::pair<double, int> p[n];
    for(int i = 0; i < n; ++i) p[i] = std::make_pair(strategy[i] / lowerbound[i], i);
    std::sort(p, p + n);

    double cur_sum = 0.0, lowerbound_sum = 0.0;
    for(int i = 0; i < n; ++i) cur_sum += strategy[i];

    /*
    u = v[idx]
    mu_sort = mu[idx]
    */

    for(int i = 0, idx, pre_idx; i < n; ++i) {
        idx = p[i].second, pre_idx = (i==0) ? 0 : p[i-1].second;

        double Z = cur_sum / (1.0 - gamma * lowerbound_sum);
        if((i == 0 || Z * gamma * lowerbound[pre_idx] >= strategy[pre_idx]) && Z * gamma * lowerbound[idx] <= strategy[idx]) {
            strategy = strategy / Z;
            double dis_sum = 0.0;
            for(int j = 0; j < n; ++j) {
                strategy[j] = (strategy[j] <= gamma * lowerbound[j]) ? gamma * lowerbound[j] 
                                                                        : strategy[j];
                dis_sum += strategy[j];
            }
            if(std::fabs(dis_sum - 1.0) > Constants::EPS)
                throw std::invalid_argument("EntropyMax: The sum of the strategy is not 1.0");
            return;
        }

        cur_sum -= strategy[idx];
        lowerbound_sum += lowerbound[idx];
    }

    strategy = lowerbound;
}

void ProjectionOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    /*
        inputs[0] is the vector to be projected
        inputs[1] is a 1*1 vector, the \gamma. Default is 0
        inputs[2] has the same size as inputs[0], the lowerbound. Default is uniform distribution

        inputs[0] will be projected to the probability simplex with lowerbound \gamma * inputs[2]
    */
    if (inputs.size() == 0) {
        throw std::invalid_argument("Projection requires at least one input");
    }

    if (inputs.size() > 3) {
        throw std::invalid_argument("Projection requires at most three inputs: vector to be projected, gamma, and lowerbound");
    }

    double gamma = (inputs.size() > 1) ? (*inputs[1])[0] : 0.0;
    if(inputs.size() > 2)
        lowerbound = *inputs[2];
    else{
        lowerbound.Resize(inputs[0]->size);
        lowerbound.Set(1.0 / double(inputs[0]->size));
    }
    
    result = *inputs[0];
    if(result.size == 0) return;

    if(distance_name == "L2") {
        SparseMax(result, gamma, lowerbound);
    } else {
        EntropyMax(result, gamma, lowerbound);
    }
}