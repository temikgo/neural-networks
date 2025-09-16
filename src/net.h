#pragma once

#include "data_loader.h"
#include "layer.h"
#include "linalg.h"
#include "loss_function.h"
#include "optimizer.h"

namespace nn {

struct EpochMetrics {
    std::vector<Scalar> losses;
    std::vector<Scalar> accuracies;

    Scalar AvgLoss() const;
    Scalar AvgAccuracy() const;
};

using TrainingHistory = std::vector<EpochMetrics>;

class Net {
public:
    Net(std::vector<Layer>&& layers);

    TrainingHistory Train(DataLoader& loader, size_t epochs,
                          const LossFunction& lossFunction,
                          const Optimizer& optimizer);
    Vector Predict(const Vector& x) const;
    Matrix Predict(const Matrix& xBatch) const;

private:
    bool IsCorrect(const std::vector<Layer>& layers) const;

    static std::vector<int> MatrixToLabelVector(const Matrix& m);

    void TrainOneEpoch(DataLoader& loader, const LossFunction& lossFunction,
                       const Optimizer& optimizer,
                       std::vector<Matrix>& layerOutputs,
                       EpochMetrics* epochMetrics);
    void TrainOnBatch(const Batch& batch, const LossFunction& lossFunction,
                      const Optimizer& optimizer,
                      std::vector<Matrix>& layerOutputs,
                      EpochMetrics* epochMetrics);

    std::vector<Layer> layers_;
};

}  // namespace nn
