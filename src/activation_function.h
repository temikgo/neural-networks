#pragma once

#include <functional>
#include <vector>

#include "helpers.h"

namespace nn {

class ActivationFunction {
    using FuncType = std::function<Number(Number)>;

public:
    ActivationFunction() = delete;
    ActivationFunction(FuncType function, FuncType derivative)
        : function_(function), derivative_(derivative) {}
    ActivationFunction(const ActivationFunction&) = default;
    ActivationFunction(ActivationFunction&&) noexcept = default;
    ActivationFunction& operator=(const ActivationFunction&) = default;
    ActivationFunction& operator=(ActivationFunction&&) noexcept = default;
    ~ActivationFunction() = default;

    Vector EvalF(const Vector& v) const;
    Vector EvalD(const Vector& v) const;

private:
    FuncType function_;
    FuncType derivative_;
};

}  // namespace nn
