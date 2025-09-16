#pragma once

#include "linalg.h"

namespace nn {

class LossFunction {
    using PenaltyFunc = std::function<Scalar(const Matrix&, const Matrix&)>;
    using GradientFunc = std::function<Matrix(const Matrix&, const Matrix&)>;

public:
    LossFunction(PenaltyFunc&& penaltyFunc, GradientFunc&& gradientFunc);

    Scalar GetLoss(const Matrix& zBatch, const Matrix& yBatch) const;
    Matrix GetGradient(const Matrix& zBatch, const Matrix& yBatch) const;

    static LossFunction MSE();
    static LossFunction MAE();
    static LossFunction CrossEntropy(Scalar epsilon = 1e-10);

private:
    PenaltyFunc penaltyFunc_;
    GradientFunc gradientFunc_;
};

}  // namespace nn
