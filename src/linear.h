#pragma once

#include "layer.h"

namespace nn {

class Linear : public Layer {
public:
    Linear(Matrix&& A, Vector&& b);

    Matrix ForwardPass(const Matrix& x) const override;
    Matrix BackwardPass(const Matrix& x, const Matrix& u) override;

private:
    Matrix A_;
    Vector b_;
};

}  // namespace nn
