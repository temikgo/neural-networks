#include "loss_function.h"

namespace nn {

LossFunction::LossFunction(PenaltyFunc&& penaltyFunc,
                           GradientFunc&& gradientFunc)
    : penaltyFunc_(std::move(penaltyFunc)),
      gradientFunc_(std::move(gradientFunc)) {
}

Scalar LossFunction::GetLoss(const Matrix& zBatch, const Matrix& yBatch) const {
    assert(penaltyFunc_ && "Loss function (forward) is not initialized");
    assert(zBatch.rows() == yBatch.rows() && zBatch.cols() == yBatch.cols());
    return penaltyFunc_(zBatch, yBatch) / static_cast<Scalar>(zBatch.cols());
}

Matrix LossFunction::GetGradient(const Matrix& zBatch,
                                 const Matrix& yBatch) const {
    assert(gradientFunc_ && "Loss function (backward) is not initialized");
    assert(zBatch.rows() == yBatch.rows() && zBatch.cols() == yBatch.cols());
    return gradientFunc_(zBatch, yBatch);
}

namespace {

Scalar MSEPenalty(const Matrix& zBatch, const Matrix& yBatch) {
    return (zBatch - yBatch).array().square().sum();
}

Matrix MSEGradient(const Matrix& zBatch, const Matrix& yBatch) {
    return (2.0 / static_cast<Scalar>(zBatch.rows())) * (zBatch - yBatch);
}

Scalar MAEPenalty(const Matrix& zBatch, const Matrix& yBatch) {
    return (zBatch - yBatch).array().abs().sum();
}

Matrix MAEGradient(const Matrix& zBatch, const Matrix& yBatch) {
    return (zBatch - yBatch).array().sign().matrix() /
           static_cast<Scalar>(zBatch.rows());
}

}  // namespace

LossFunction LossFunction::MSE() {
    return LossFunction(MSEPenalty, MSEGradient);
}

LossFunction LossFunction::MAE() {
    return LossFunction(MAEPenalty, MAEGradient);
}

LossFunction LossFunction::CrossEntropy(Scalar epsilon) {
    return LossFunction(
        [epsilon](const Matrix& zBatch, const Matrix& yBatch) {
            return -(yBatch.array() * (zBatch.array() + epsilon).log()).sum();
        },
        [epsilon](const Matrix& zBatch, const Matrix& yBatch) {
            return (-(yBatch.array() / (zBatch.array() + epsilon))).matrix();
        });
}

}  // namespace nn
