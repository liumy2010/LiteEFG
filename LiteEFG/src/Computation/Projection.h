#ifndef PROJECTION_H_
#define PROJECTION_H_

#include "Operations.h"
#include "Data/Vector.h"

#include <string>

class ProjectionOperation : public Operation {
private:
    Vector lowerbound;
public:
    std::string distance_name;
    ProjectionOperation(const std::string &distance_name_, const bool& is_static_=false);

    void Execute(Vector& result, const std::vector<Vector*>& inputs) override;
};

#endif