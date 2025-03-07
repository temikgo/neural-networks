#pragma once

#include "linalg.h"

namespace nn {

class Layer {
public:
    virtual Matrix ForwardPass(const Matrix& x) const = 0;
    virtual Matrix BackwardPass(const Matrix& x, const Matrix& u) = 0;
};

}  // namespace nn
