#pragma once

#include "linalg.h"

namespace nn {

class LossFunction {
    using DistFuncType = std::function<Vector(const Matrix&, const Matrix&)>;
    using GradFuncType = std::function<Matrix(const Matrix&, const Matrix&)>;

public:
    LossFunction(DistFuncType&& distFunc, GradFuncType&& gradFunc);

    Vector Dist(const Matrix& x, const Matrix& y) const;
    Matrix Grad(const Matrix& x, const Matrix& y) const;

    static LossFunction MSE() {
        return LossFunction([](const Matrix& x, const Matrix& y) { return x; },
                            [](const Matrix& x, const Matrix& y) { return x; });
    }

private:
    DistFuncType distFunc_;
    GradFuncType gradFunc_;
};

}  // namespace nn
