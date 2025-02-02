#include "activation_function.h"

#include <algorithm>

namespace nn {

Vector ActivationFunction::EvalF(const Vector& v) const {
    Vector evaluated(v.size());
    std::transform(v.begin(), v.end(), evaluated.begin(), function_);
    return evaluated;
}

Vector ActivationFunction::EvalD(const Vector& v) const {
    Vector evaluated(v.size());
    std::transform(v.begin(), v.end(), evaluated.begin(), derivative_);
    return evaluated;
}

}  // namespace nn
