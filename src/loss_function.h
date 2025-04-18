#pragma once

#include "linalg.h"

namespace nn {

class LossFunction {
    using PenaltyFunc = std::function<Scalar(const Vector&, const Vector&)>;
    using GradientFunc = std::function<Vector(const Matrix&, const Matrix&)>;

public:
    LossFunction(PenaltyFunc&& penaltyFunc, GradientFunc&& gradientFunc);

    Scalar GetPenalty(const Vector& z, const Vector& y) const;
    Scalar GetLoss(const VectorBatch& z, const VectorBatch& y) const;
    Vector GetGradient(const Vector& z, const Vector& y) const;

    static LossFunction MSE();
    static LossFunction MAE();
    static LossFunction CrossEntropy();

private:
    PenaltyFunc penaltyFunc_;
    GradientFunc gradientFunc_;
};

}  // namespace nn
