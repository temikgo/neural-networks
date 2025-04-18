#include "layer.h"

#include "utils.h"

namespace nn {

Layer::Layer(Index inputDim, Index outputDim)
    : Layer(inputDim, outputDim, GenerateMatrix(outputDim, inputDim),
            GenerateVector(outputDim), ActivationFunction::Id()) {
}

Layer::Layer(Index inputDim, Index outputDim, ActivationFunction&& sigma)
    : Layer(inputDim, outputDim, GenerateMatrix(outputDim, inputDim),
            GenerateVector(outputDim), std::move(sigma)) {
}

Layer::Layer(Index inputDim, Index outputDim, Matrix&& A, Vector&& b)
    : Layer(inputDim, outputDim, std::move(A), std::move(b),
            ActivationFunction::Id()) {
}

Layer::Layer(Index inputDim, Index outputDim, Matrix&& A, Vector&& b,
             ActivationFunction&& sigma)
    : A_(std::move(A)), b_(std::move(b)), sigma_(std::move(sigma)) {
    A_.normalize();
    b_.normalize();
}

Index Layer::GetInputDim() const {
    return A_.cols();
}

Index Layer::GetOutputDim() const {
    return A_.rows();
}

void Layer::ZeroGradients() {
    AGrad_ = Matrix::Zero(GetOutputDim(), GetInputDim());
    bGrad_ = Vector::Zero(GetOutputDim());
}

Vector Layer::ForwardPass(const Vector& x) const {
    return sigma_.Compute(A_ * x + b_);
}

Vector Layer::BackwardPass(const Vector& x, const Vector& y, const Vector& u,
                           size_t batchSize) {
    Matrix jacobian = sigma_.ComputeGradient(y);

    Matrix AGrad = (jacobian * u) * x.transpose();
    Vector bGrad = jacobian * u;

    AGrad_ += AGrad / batchSize;
    bGrad_ += bGrad / batchSize;

    return A_.transpose() * (jacobian * u);
}

void Layer::Update(std::function<void(Matrix&, const Matrix&)> AUpdater,
                   std::function<void(Vector&, const Vector&)> bUpdater) {
    AUpdater(A_, AGrad_);
    bUpdater(b_, bGrad_);
}

}  // namespace nn
