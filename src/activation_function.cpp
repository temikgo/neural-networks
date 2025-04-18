#include "activation_function.h"

#include "src/linalg.h"

namespace nn {

ActivationFunction::ActivationFunction(ForwardFunc&& function,
                                       BackwardFunc&& derivative)
    : function_(std::move(function)), derivative_(std::move(derivative)) {
}

Vector ActivationFunction::Compute(const Vector& x) const {
    assert(function_ && "Activation function (forward) is not initialized");
    return function_(x);
}

Matrix ActivationFunction::ComputeGradient(const Vector& x) const {
    assert(derivative_ && "Activation function (backward) is not initialized");
    return derivative_(x);
}

static Vector IdFunction(const Vector& x) {
    return x;
}

static Matrix IdDerivative(const Vector& x) {
    return Matrix::Identity(x.size(), x.size());
}

static Vector SigmoidFunction(const Vector& x) {
    return x.unaryExpr([](Scalar xi) { return 1 / (1 + exp(-xi)); });
}

static Matrix SigmoidDerivative(const Vector& x) {
    Vector y = SigmoidFunction(x);
    return (y.cwiseProduct(Vector::Ones(x.size()) - y)).asDiagonal();
}

static Vector TanhFunction(const Vector& x) {
    return x.unaryExpr([](Scalar xi) { return tanh(xi); });
}

static Matrix TanhDerivative(const Vector& x) {
    Vector y = TanhFunction(x);
    Vector l = Vector::Ones(x.size()) - y;
    Vector r = Vector::Ones(x.size()) + y;
    return (l.cwiseProduct(r)).asDiagonal();
}

static Vector ReLUFunction(const Vector& x) {
    return x.cwiseMax(0);
}

static Matrix ReLUDerivative(const Vector& x) {
    return Matrix((x.array() > 0).cast<Scalar>().matrix().asDiagonal());
}

static Vector SoftmaxFunction(const Vector& x) {
    Vector exps = (x.array() - x.maxCoeff()).exp();
    return exps / exps.sum();
}

static Matrix SoftmaxDerivative(const Vector& x) {
    Vector y = SoftmaxFunction(x);
    return Matrix(y.asDiagonal()) - y * y.transpose();
}

ActivationFunction ActivationFunction::Id() {
    return ActivationFunction(IdFunction, IdDerivative);
}

ActivationFunction ActivationFunction::Sigmoid() {
    return ActivationFunction(SigmoidFunction, SigmoidDerivative);
}

ActivationFunction ActivationFunction::Tanh() {
    return ActivationFunction(TanhFunction, TanhDerivative);
}

ActivationFunction ActivationFunction::ReLU() {
    return ActivationFunction(ReLUFunction, ReLUDerivative);
}

ActivationFunction ActivationFunction::Softmax() {
    return ActivationFunction(SoftmaxFunction, SoftmaxDerivative);
}

}  // namespace nn
