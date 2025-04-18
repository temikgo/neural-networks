#include "net.h"

namespace nn {

Net::Net(std::vector<Layer>&& layers) {
    assert(!layers.empty() && "Network layers are not initialized");
    for (size_t i = 1; i < layers.size(); ++i) {
        assert(layers[i - 1].GetOutputDim() == layers[i].GetInputDim() &&
               "Layer dimension mismatch between consecutive layers");
    }
    layers_ = std::move(layers);
}

void Net::Train(const VectorBatch& xBatch, const VectorBatch& yBatch,
                size_t batchSize, size_t epochs,
                const LossFunction& loss_function, const Optimizer& optimizer) {
    assert(batchSize > 0 && "Batch size must be positive");
    assert(epochs > 0 && "Number of epochs must be positive");
    assert(xBatch.size() == yBatch.size() && "x and y must be same size");
    for (size_t i = 0; i < epochs; ++i) {
        size_t actualBatchSize;
        for (size_t j = 0; j < xBatch.size(); ++j) {
            if (j % batchSize == 0) {
                actualBatchSize = std::min(batchSize, xBatch.size() - j);
                ZeroGradients();
            }
            Fit(xBatch[j], yBatch[j], actualBatchSize, loss_function,
                optimizer);
            if (j % batchSize == actualBatchSize - 1) {
                optimizer.DoStep(layers_);
            }
        }
    }
}

void Net::Fit(const Vector& x, const Vector& y, size_t batchSize,
              const LossFunction& loss_function, const Optimizer& optimizer) {
    std::vector<Vector> layerOutputs;
    layerOutputs.reserve(layers_.size() + 1);
    layerOutputs.push_back(x);

    for (const Layer& layer : layers_) {
        layerOutputs.push_back(layer.ForwardPass(layerOutputs.back()));
    }

    Vector grad = loss_function.GetGradient(layerOutputs.back(), y);

    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad = layers_[i].BackwardPass(layerOutputs[i], layerOutputs[i + 1],
                                       grad, batchSize);
    }
}

Vector Net::Predict(const Vector& x) const {
    Vector y = x;
    for (const Layer& layer : layers_) {
        y = layer.ForwardPass(y);
    }
    return y;
}

VectorBatch Net::Predict(const VectorBatch& xBatch) const {
    VectorBatch yBatch;
    yBatch.reserve(xBatch.size());
    for (const Vector& x : xBatch) {
        yBatch.push_back(Predict(x));
    }
    return yBatch;
}

void Net::ZeroGradients() {
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].ZeroGradients();
    }
}

}  // namespace nn
