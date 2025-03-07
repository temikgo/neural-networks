#pragma once

#include <functional>

#include "layer.h"

namespace nn {

class ActivationFunction : public Layer {
    using FuncType = std::function<double(double)>;

public:
    ActivationFunction(FuncType&& function, FuncType&& derivative);

    Matrix ForwardPass(const Matrix& x) const override;
    Matrix BackwardPass(const Matrix& x, const Matrix& u) override;

private:
    FuncType function_;
    FuncType derivative_;
};

}  // namespace nn
