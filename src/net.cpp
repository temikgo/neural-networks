#include "net.h"

#include "utils.h"

namespace nn {

Scalar EpochMetrics::AvgLoss() const {
    if (losses.empty()) {
        return 0;
    }
    return std::accumulate(losses.begin(), losses.end(), Scalar(0)) /
           losses.size();
}

Scalar EpochMetrics::AvgAccuracy() const {
    if (accuracies.empty()) {
        return 0;
    }
    return std::accumulate(accuracies.begin(), accuracies.end(), Scalar(0)) /
           accuracies.size();
}

bool Net::IsCorrect(const std::vector<Layer>& layers) const {
    if (layers.empty()) {
        return false;
    }
    for (size_t i = 1; i < layers.size(); ++i) {
        if (layers[i - 1].GetOutputDim() != layers[i].GetInputDim()) {
            return false;
        }
    }
    return true;
}

std::vector<int> Net::MatrixToLabelVector(const Matrix& m) {
    const int cols = static_cast<int>(m.cols());
    std::vector<int> labels;
    labels.reserve(cols);

    for (int j = 0; j < cols; ++j) {
        Eigen::Index idx;
        m.col(j).maxCoeff(&idx);
        labels.push_back(static_cast<int>(idx));
    }

    return labels;
}

Net::Net(std::vector<Layer>&& layers) : layers_(std::move(layers)) {
    assert(IsCorrect(layers_) && "Incorrect layer structure.");
}

TrainingHistory Net::Train(DataLoader& loader, size_t epochs,
                           const LossFunction& loss_function,
                           const Optimizer& optimizer) {
    assert(epochs > 0 && "Number of epochs must be positive");
    std::vector<Matrix> layerOutputs;
    layerOutputs.reserve(layers_.size() + 1);

    TrainingHistory history;
    history.reserve(epochs);

    for (size_t i = 0; i < epochs; ++i) {
        EpochMetrics epochMetrics;
        TrainOneEpoch(loader, loss_function, optimizer, layerOutputs,
                      &epochMetrics);
        history.push_back(std::move(epochMetrics));
    }

    return history;
}

void Net::TrainOneEpoch(DataLoader& loader, const LossFunction& loss_function,
                        const Optimizer& optimizer,
                        std::vector<Matrix>& layerOutputs,
                        EpochMetrics* epochMetrics) {
    loader.Permute();
    for (auto batch = loader.begin(); batch < loader.end(); ++batch) {
        TrainOnBatch(*batch, loss_function, optimizer, layerOutputs,
                     epochMetrics);
    }
}

void Net::TrainOnBatch(const Batch& batch, const LossFunction& loss_function,
                       const Optimizer& optimizer,
                       std::vector<Matrix>& layerOutputs,
                       EpochMetrics* epochMetrics) {
    layerOutputs.clear();
    layerOutputs.push_back(batch.x);

    for (const Layer& layer : layers_) {
        layerOutputs.push_back(layer.ForwardPass(layerOutputs.back()));
    }

    if (epochMetrics) {
        Scalar batchLoss = loss_function.GetLoss(layerOutputs.back(), batch.y);
        Scalar batchAcc = Accuracy(MatrixToLabelVector(layerOutputs.back()),
                                   MatrixToLabelVector(batch.y));
        epochMetrics->losses.push_back(batchLoss);
        epochMetrics->accuracies.push_back(batchAcc);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].ZeroGradients();
    }

    Matrix grad = loss_function.GetGradient(layerOutputs.back(), batch.y);

    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad =
            layers_[i].BackwardPass(layerOutputs[i], layerOutputs[i + 1], grad);
    }
    optimizer.DoStep(layers_);
}

Vector Net::Predict(const Vector& x) const {
    Vector y = x;
    for (const Layer& layer : layers_) {
        y = layer.ForwardPass(y);
    }
    return y;
}

Matrix Net::Predict(const Matrix& xBatch) const {
    Matrix yBatch = xBatch;
    for (const Layer& layer : layers_) {
        yBatch = layer.ForwardPass(yBatch);
    }
    return yBatch;
}

}  // namespace nn
