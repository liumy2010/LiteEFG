#include "Operations.h"

#include "Basic/Constants.h"
#include "Basic/BasicFunction.h"

#include <stdexcept>
#include <cmath>

void CopyOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Copy only supports one input");
    }
    
    result = *inputs[0];
}

void AddOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() == 0) {
        throw std::invalid_argument("Sum requires at least one input");
    }
    
    tmp = *inputs[0];
    for (int i = 1; i < inputs.size(); ++i) {
        tmp.Add(*inputs[i]);
    }
    result = tmp;
}

void SubOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Sub requires only two inputs");
    }

    tmp = *inputs[0];
    tmp.Sub(*inputs[1]);
    result = tmp;
}

void MulOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() == 0) {
        throw std::invalid_argument("Mul requires at least one inputs");
    }

    tmp = *inputs[0];
    for (int i = 1; i < inputs.size(); ++i) {
        tmp.Mul(*inputs[i]);
    }
    result = tmp;
}

void DivOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Div requires two inputs");
    }

    tmp = *inputs[0];
    tmp.Div(*inputs[1]);
    result = tmp;
}

void ExpOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Exp requires only one input");
    }

    result.Resize(inputs[0]->size);
    for(int i = 0; i < result.size; ++i) {
        result[i] = std::exp((*inputs[0])[i]);
    }
}

void LogOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Log requires only one input");
    }

    result.Resize(inputs[0]->size);
    for(int i = 0; i < result.size; ++i) {
        double x = (*inputs[0])[i];
        if(x <= - Constants::EPS) {
            throw std::invalid_argument("Log requires all elements to be positive");
        }
        result[i] = (x < Constants::EPS) ? std::log(Constants::EPS) : std::log(x);
    }
}

void SumOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Sum requires only one input");
    }

    double sum = 0.0;
    for(int i = 0; i < inputs[0]->size; ++i) {
        sum += (*inputs[0])[i];
    }
    result.Resize(1);
    result[0] = sum;
}

void MeanOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Mean requires only one input");
    }

    double sum = 0.0;
    for(int i = 0; i < inputs[0]->size; ++i) {
        sum += (*inputs[0])[i];
    }
    result.Resize(1);
    result[0] = (inputs[0]->size==0) ? 0.0 : sum / double(inputs[0]->size);
}

void MaxOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Max requires only one input");
    }

    double maximum = -Constants::INF;
    for(int i = 0; i < inputs[0]->size; ++i) {
        maximum = std::max(maximum, (*inputs[0])[i]);
    }
    result.Resize(1);
    result[0] = maximum;
}

void MinOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Min requires only one input");
    }

    double minimum = Constants::INF;
    for(int i = 0; i < inputs[0]->size; ++i) {
        minimum = std::min(minimum, (*inputs[0])[i]);
    }
    result.Resize(1);
    result[0] = minimum;
}

void DotOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Dot requires only two inputs");
    }
    if(inputs[0]->size != inputs[1]->size) {
        throw std::invalid_argument("Dot requires both inputs to have the same size");
    }

    double sum = 0.0;
    for(int i = 0; i < inputs[0]->size; ++i) {
        sum += (*inputs[0])[i] * (*inputs[1])[i];
    }
    result.Resize(1);
    result[0] = sum;
}

void ArgmaxOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Argmax requires only one input");
    }

    int argmax = 0, maximum = -Constants::INF;
    for(int i = 1; i < inputs[0]->size; ++i) {
        if((*inputs[0])[i] > maximum){
            argmax = i;
            maximum = (*inputs[0])[i];
        }
    }
    result.Resize(inputs[0]->size);
    result.Set(0.0);
    result[argmax] = 1.0;
}

void ArgminOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Argmin requires only one input");
    }

    int argmin = 0, minimum = Constants::INF;
    for(int i = 1; i < inputs[0]->size; ++i) {
        if((*inputs[0])[i] < minimum){
            argmin = i;
            minimum = (*inputs[0])[i];
        }
    }
    result.Resize(inputs[0]->size);
    result.Set(0.0);
    result[argmin] = 1.0;
}

void MaximumOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Maximum requires only two inputs");
    }
    if(inputs[0]->size != inputs[1]->size && inputs[0]->size != 1 && inputs[1]->size != 1) {
        throw std::invalid_argument("Maximum requires both inputs to have the same size or one of them is of size 1");
    }

    tmp.Resize(std::max(inputs[0]->size, inputs[1]->size));
    for(int i = 0; i < tmp.size; ++i) {
        tmp[i] = (*inputs[0])[(inputs[0] -> size == 1)?0:i];
        tmp[i] = std::max(tmp[i], (*inputs[1])[(inputs[1]->size==1)?0:i]);
    }
    result = tmp;
}

void MinimumOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Minimum requires only two inputs");
    }
    if(inputs[0]->size != inputs[1]->size && inputs[0]->size != 1 && inputs[1]->size != 1) {
        throw std::invalid_argument("Minimum requires both inputs to have the same size or one of them is of size 1");
    }

    tmp.Resize(std::max(inputs[0]->size, inputs[1]->size));
    for(int i = 0; i < tmp.size; ++i) {
        tmp[i] = (*inputs[0])[(inputs[0] -> size == 1)?0:i];
        tmp[i] = std::min(tmp[i], (*inputs[1])[(inputs[1]->size==1)?0:i]);
    }
    result = tmp;
}

void EuclideanOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if(inputs.size() != 1) {
        throw std::invalid_argument("Euclidean requires only one input");
    }

    double sum=0;
    for(int i = 0; i < inputs[0]->size; ++i) {
        sum += Basic::Sqr((*inputs[0])[i]);
    }
    result.Resize(1);
    result[0] = 0.5 * sum;
}

void NegativeEntropyOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if(inputs.size() != 1) {
        throw std::invalid_argument("Entropy requires only one input");
    }

    double sum = (shifted)?std::log(double(inputs[0]->size)):0.0; // shifted entropy so that the minimum value is 0
    for(int i = 0; i < inputs[0]->size; ++i) {
        if((*inputs[0])[i] < -Constants::EPS) {
            throw std::invalid_argument("Entropy requires all elements to be non-negative");
        }
        sum += (*inputs[0])[i] * std::log((*inputs[0])[i]);
    }
    result.Resize(1); 
    result[0] = sum;
}

AggregateOperation::AggregateOperation(std::shared_ptr<Operation> aggregator_, const std::string& object, const std::string& player, const double& padding, const bool& is_static_)
                                        : Operation("Aggregate", is_static_), aggregator{aggregator_} {
    if(aggregator -> name != "Max" && aggregator -> name != "Min" && aggregator -> name != "Sum") {
        throw std::invalid_argument("Aggregator must be Max, Min, Sum or Mean");
    }
    if(object != "children" && object != "parent") {
        throw std::invalid_argument("Aggregate object must be children or parent");
    }
    if(player != "self" && player != "opponents") {
        throw std::invalid_argument("Player to be aggregated must be self or opponents");
    }
    info = Vector(3, 0.0);
    info[AggregateOperation::InfoIndex::padding] = padding;
    info[AggregateOperation::InfoIndex::object] = (object == "children") ? 1.0 : -1.0;
    info[AggregateOperation::InfoIndex::player] = (player == "self") ? 1.0 : -1.0;
}

void AggregateOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    result.Resize(inputs.size()); // Since result will never be one of the inputs for aggregator operation, it is fine here
    for(int i = 0; i < inputs.size(); ++i) {
        if(inputs[i] -> size == 0){
            result[i] = info[AggregateOperation::InfoIndex::padding];
            continue;
        }
        aggregator -> Execute(tmp, {inputs[i]}); // Maybe inefficient {}
        result[i] = tmp[0]; // It was modified by the aggregator, since aggregator only output a scalar
    }
}

NormalizeOperation::NormalizeOperation(const double& p_norm_, const bool& ignore_negative_, const bool& is_static_)
                                                : Operation("Normalize", is_static_), p_norm{p_norm_} 
                                                    , ignore_negative{ignore_negative_} {
    if(p_norm < -Constants::EPS){
        throw std::invalid_argument("Normalize requires the p-norm to be positive");
    }
}

void NormalizeOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Normalize requires only one input");
    }

    result = *inputs[0];
    if(ignore_negative) {
        for(int i = 0; i < result.size; ++i) {
            result[i] = (result[i] < 0.0) ? 0.0 : result[i];
        }
    }
    if(p_norm < Constants::EPS) {
        result.Div(double(result.size));
        return;
    }
    double sum = 0.0;
    for(int i = 0; i < result.size; ++i) {
        sum += pow(fabs(result[i]), p_norm);
    }
    if(sum < Constants::EPS) {
        result.Set(1.0 / double(result.size));
        return;
    }
    sum = pow(sum, 1.0 / p_norm);
    for(int i = 0; i < result.size; ++i) {
        result[i] /= sum;
    }
}

CompareOperation::CompareOperation(const int& type_, const bool& is_static_) : Operation("Compare", is_static_), type{type_} {
    if(type < ComparatorType::start || type >= ComparatorType::type_num) {
        throw std::invalid_argument("Compare requires the type to be in the range [0, 4]");
    }
}

void CompareOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Compare requires only two inputs");
    }
    if(inputs[0]->size != inputs[1]->size && inputs[0]->size != 1 && inputs[1]->size != 1) {
        throw std::invalid_argument("Compare requires both inputs to have the same size or one of them is a scalar");
    }

    int size0 = inputs[0]->size, size1 = inputs[1]->size, size = std::max(size0, size1);
    result.Resize(size);
    switch(type) {
        case ComparatorType::greater_than:
            for(int i = 0; i < size; ++i) {
                result[i] = ((*inputs[0])[(size0==1)?0:i] > (*inputs[1])[(size1==1)?0:i]) ? 1.0 : 0.0;
            }
            break;
        case ComparatorType::greater_than_or_equal:
            for(int i = 0; i < size; ++i) {
                result[i] = ((*inputs[0])[(size0==1)?0:i] >= (*inputs[1])[(size1==1)?0:i]) ? 1.0 : 0.0;
            }
            break;
        case ComparatorType::less_than:
            for(int i = 0; i < size; ++i) {
                result[i] = ((*inputs[0])[(size0==1)?0:i] < (*inputs[1])[(size1==1)?0:i]) ? 1.0 : 0.0;
            }
            break;
        case ComparatorType::less_than_or_equal:
            for(int i = 0; i < size; ++i) {
                result[i] = ((*inputs[0])[(size0==1)?0:i] <= (*inputs[1])[(size1==1)?0:i]) ? 1.0 : 0.0;
            }
            break;
        case ComparatorType::equal:
            for(int i = 0; i < size; ++i) {
                result[i] = (fabs((*inputs[0])[(size0==1)?0:i] - (*inputs[1])[(size1==1)?0:i])<Constants::EPS) ? 1.0 : 0.0;
            }
            break;
        default:
            throw std::invalid_argument("Compare requires the type to be in the range [0, 4]");
    }
}

void PowOperation::Execute(Vector& result, const std::vector<Vector*>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Pow requires only two inputs");
    }
    if(inputs[1]->size != 1) {
        throw std::invalid_argument("Pow requires the second input to be a scalar");
    }

    result = *inputs[0];
    double power = (*inputs[1])[0];
    for(int i = 0; i < result.size; ++i) {
        result[i] = pow(result[i], power);
    }
}