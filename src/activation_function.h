#pragma once

#include "linalg.h"

namespace nn {

class ActivationFunction {
    using ForwardFunc = std::function<Matrix(const Matrix&)>;
    using BackwardFunc = std::function<Matrix(const Matrix&, const Matrix&)>;

public:
    ActivationFunction(ForwardFunc&& function, BackwardFunc&& derivative);
    Matrix Forward(const Matrix& xBatch) const;
    Matrix Backward(const Matrix& yBatch, const Matrix& uBatch) const;

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
