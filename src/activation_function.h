#pragma once

#include "linalg.h"

namespace nn {

class ActivationFunction {
    using ForwardFunc = std::function<Vector(const Vector&)>;
    using BackwardFunc = std::function<Matrix(const Vector&)>;

public:
    ActivationFunction(ForwardFunc&& function, BackwardFunc&& derivative);
    Vector Compute(const Vector& x) const;
    Matrix ComputeGradient(const Vector& x) const;

    static ActivationFunction Id();
    static ActivationFunction Sigmoid();
    static ActivationFunction Tanh();
    static ActivationFunction ReLU();
    static ActivationFunction Softmax();

private:
    ForwardFunc function_;
    BackwardFunc derivative_;
};

}  // namespace nn
