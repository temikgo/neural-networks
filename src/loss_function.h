#pragma once

#include <vector>

#include "helpers.h"

namespace nn {

class LossFunction {
public:
    virtual Number Dist(const Vector& x, const Vector& y) const = 0;
    virtual Number Calc(const Vector& x, const Vector& y) const = 0;
};

class MSELoss : public LossFunction {
    Number Dist(const Vector& x, const Vector& y) const override final;
    Number Calc(const Vector& x, const Vector& y) const override final;
};

class MAELoss : public LossFunction {
    Number Dist(const Vector& x, const Vector& y) const override final;
    Number Calc(const Vector& x, const Vector& y) const override final;
};

}  // namespace nn
