#include "loss_function.h"

namespace nn {

LossFunction::LossFunction(DistFuncType&& distFunc, GradFuncType&& gradFunc)
    : distFunc_(std::move(distFunc)), gradFunc_(std::move(gradFunc)) {}

Vector LossFunction::Dist(const Matrix& x, const Matrix& y) const {
    return distFunc_(x, y);
}

Matrix LossFunction::Grad(const Matrix& x, const Matrix& y) const {
    return gradFunc_(x, y);
}

}  // namespace nn
