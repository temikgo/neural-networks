#include "layer.h"

#include "utils.h"

namespace nn {

Layer::Layer(In inputDim, Out outputDim, ActivationFunction&& sigma)
    : Layer(GenerateMatrix(outputDim, inputDim), GenerateVector(outputDim),
            std::move(sigma)) {
}

Layer::Layer(Matrix&& A, Vector&& b, ActivationFunction&& sigma)
    : A_(std::move(A)), b_(std::move(b)), sigma_(std::move(sigma)) {
    assert(A_.rows() == b_.size() && "Layer::A and Layer::B size mismatch.");
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

Matrix Layer::ForwardPass(const Matrix& xBatch) const {
    return sigma_.Forward((A_ * xBatch).colwise() + b_);
}

Matrix Layer::BackwardPass(const Matrix& xBatch, const Matrix& yBatch,
                           const Matrix& uBatch) {
    Matrix delta = sigma_.Backward(yBatch, uBatch);

    AGrad_ += (delta * xBatch.transpose()) / xBatch.cols();
    bGrad_ += delta.rowwise().sum() / xBatch.cols();

    return A_.transpose() * delta;
}

void Layer::Update(std::function<void(Matrix&, const Matrix&)> AUpdater,
                   std::function<void(Vector&, const Vector&)> bUpdater) {
    AUpdater(A_, AGrad_);
    bUpdater(b_, bGrad_);
}

}  // namespace nn
