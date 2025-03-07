#include "activation_function.h"

namespace nn {

ActivationFunction::ActivationFunction(FuncType&& function,
                                       FuncType&& derivative)
    : function_(std::move(function)), derivative_(std::move(derivative)) {}

Matrix ActivationFunction::ForwardPass(const Matrix& x) const {
    assert(function_);
    return x.unaryExpr(function_);
}

Matrix ActivationFunction::BackwardPass(const Matrix& x, const Matrix& u) {
    assert(derivative_);
    return x.unaryExpr(derivative_);
}

}  // namespace nn
