#include "builder.h"

namespace nn {

Builder& Builder::Input(Index inputSize) {
    assert(inputSize > 0 && "Input size must be positive.");
    inputSize_ = inputSize;
    return *this;
}

Builder& Builder::AddLayer(Index outputSize, ActivationFunction activation) {
    assert(inputSize_ > 0 && "Input isn't set.");
    assert(outputSize > 0 && "Output size must be positive.");
    layers_.emplace_back(In{inputSize_}, Out{outputSize},
                         std::move(activation));
    inputSize_ = outputSize;
    return *this;
}

Net Builder::Extract() {
    assert(!layers_.empty() && "No layers added.");
    return Net(std::move(layers_));
}

}  // namespace nn
