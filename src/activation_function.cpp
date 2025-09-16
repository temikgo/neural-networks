#include "activation_function.h"

namespace nn {

ActivationFunction::ActivationFunction(ForwardFunc&& function,
                                       BackwardFunc&& derivative)
    : function_(std::move(function)), derivative_(std::move(derivative)) {
}

Matrix ActivationFunction::Forward(const Matrix& xBatch) const {
    assert(function_ && "Activation function (forward) is not initialized");
    return function_(xBatch);
}

Matrix ActivationFunction::Backward(const Matrix& yBatch,
                                    const Matrix& uBatch) const {
    assert(derivative_ && "Activation function (backward) is not initialized");
    return derivative_(yBatch, uBatch);
}

namespace {

Matrix IdFunction(const Matrix& x) {
    return x;
}

Matrix IdDerivative(const Matrix& y, const Matrix& u) {
    return u;
}

Matrix SigmoidFunction(const Matrix& x) {
    return x.unaryExpr([](Scalar xi) { return 1 / (1 + exp(-xi)); });
}

Matrix SigmoidDerivative(const Matrix& y, const Matrix& u) {
    return (y.array() * (1 - y.array())).matrix().array() * u.array();
}

Matrix TanhFunction(const Matrix& x) {
    return x.unaryExpr([](Scalar xi) { return tanh(xi); });
}

Matrix TanhDerivative(const Matrix& y, const Matrix& u) {
    return (1 - y.array().square()).matrix().array() * u.array();
}

Matrix ReLUFunction(const Matrix& x) {
    return x.cwiseMax(0);
}

Matrix ReLUDerivative(const Matrix& y, const Matrix& u) {
    Matrix mask = (y.array() > Scalar(0)).cast<Scalar>().matrix();
    return mask.array() * u.array();
}

Matrix SoftmaxFunction(const Matrix& x) {
    const int cols = static_cast<int>(x.cols());
    const int rows = static_cast<int>(x.rows());
    Matrix x_shifted(rows, cols);
    RowVector col_max = x.colwise().maxCoeff();
    for (int j = 0; j < cols; ++j) {
        x_shifted.col(j) = x.col(j).array() - col_max(j);
    }
    Matrix exps = x_shifted.array().exp();
    RowVector sums = exps.colwise().sum();
    Matrix out(rows, cols);
    for (int j = 0; j < cols; ++j) {
        out.col(j) = exps.col(j).array() / sums(j);
    }
    return out;
}

Matrix SoftmaxDerivative(const Matrix& y, const Matrix& u) {
    const int cols = static_cast<int>(y.cols());
    const int rows = static_cast<int>(y.rows());
    RowVector dots = (y.array() * u.array()).colwise().sum();
    Matrix tmp(rows, cols);
    for (int j = 0; j < cols; ++j) {
        tmp.col(j) = u.col(j).array() - dots(j);
    }
    return y.array() * tmp.array();
}

}  // namespace

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
