#include "loss_function.h"

namespace nn {

LossFunction::LossFunction(PenaltyFunc&& penaltyFunc,
                           GradientFunc&& gradientFunc)
    : penaltyFunc_(std::move(penaltyFunc)),
      gradientFunc_(std::move(gradientFunc)) {
}

Scalar LossFunction::GetPenalty(const Vector& z, const Vector& y) const {
    assert(penaltyFunc_ && "Loss function (forward) is not initialized");
    return penaltyFunc_(z, y);
}

Scalar LossFunction::GetLoss(const VectorBatch& z, const VectorBatch& y) const {
    Scalar total = 0;
    for (size_t i = 0; i < z.size(); ++i) {
        total += GetPenalty(z[i], y[i]);
    }
    return total / z.size();
}

Vector LossFunction::GetGradient(const Vector& z, const Vector& y) const {
    assert(gradientFunc_ && "Loss function (backward) is not initialized");
    return gradientFunc_(z, y);
}

static Scalar MSEPenalty(const Vector& z, const Vector& y) {
    return (z - y).squaredNorm();
}

static Vector MSEGradient(const Vector& z, const Vector& y) {
    return 2 * (z - y) / z.size();
}

static Scalar MAEPenalty(const Vector& z, const Vector& y) {
    return (z - y).cwiseAbs().sum();
}

static Vector MAEGradient(const Vector& z, const Vector& y) {
    return Vector((z - y).array().sign()) / z.size();
}

static constexpr double eps = 1e-10;

static Scalar CrossEntropyPenalty(const Vector& z, const Vector& y) {
    return -(y.array() * (z.array() + eps).log()).sum();
}

static Vector CrossEntropyGradient(const Vector& z, const Vector& y) {
    return Vector(-(y.array() / (z.array() + eps)));
}

LossFunction LossFunction::MSE() {
    return LossFunction(MSEPenalty, MSEGradient);
}

LossFunction LossFunction::MAE() {
    return LossFunction(MAEPenalty, MAEGradient);
}

LossFunction LossFunction::CrossEntropy() {
    return LossFunction(CrossEntropyPenalty, CrossEntropyGradient);
}

}  // namespace nn
