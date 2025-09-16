#pragma once

#include "layer.h"
#include "linalg.h"

namespace nn {

class Optimizer {
public:
    Optimizer(Scalar learningRate);
    void DoStep(std::vector<Layer>& layers) const;

private:
    Scalar learningRate_;
};

}  // namespace nn
