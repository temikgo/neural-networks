#pragma once

#include "layer.h"
#include "linalg.h"
#include "loss_function.h"
#include "optimizer.h"

namespace nn {

class Net {
public:
    Net(std::vector<Layer>&& layers);

    void Train(const VectorBatch& xBatch, const VectorBatch& yBatch,
               size_t batchSize, size_t epochs,
               const LossFunction& loss_function, const Optimizer& optimizer);
    void Fit(const Vector& x, const Vector& y, size_t batchSize,
             const LossFunction& loss_function, const Optimizer& optimizer);
    Vector Predict(const Vector& x) const;
    VectorBatch Predict(const VectorBatch& xBatch) const;

    void ZeroGradients();

private:
    std::vector<Layer> layers_;
};

}  // namespace nn
