#pragma once

#include "linalg.h"
#include "net.h"

namespace nn {

class Builder {
public:
    Builder& Input(Index inputSize);

    Builder& AddLayer(Index outputSize, ActivationFunction activation);

    Net Extract();

private:
    Index inputSize_;
    std::vector<Layer> layers_;
};

}  // namespace nn
