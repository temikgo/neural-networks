#include "linear.h"

namespace nn {

Linear::Linear(Matrix&& A, Vector&& b) : A_(std::move(A)), b_(std::move(b)) {}

Matrix Linear::ForwardPass(const Matrix& x) const {
    return (A_ * x).colwise() + b_;
}

Matrix Linear::BackwardPass(const Matrix& x, const Matrix& u) { return A_; };

}  // namespace nn
