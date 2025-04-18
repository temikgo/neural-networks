#pragma once

#include "activation_function.h"
#include "linalg.h"

namespace nn {

class Layer {
public:
    Layer(Index inputDim, Index outputDim);
    Layer(Index inputDim, Index outputDim, ActivationFunction&& sigma);
    Layer(Index inputDim, Index outputDim, Matrix&& A, Vector&& b);
    Layer(Index inputDim, Index outputDim, Matrix&& A, Vector&& b,
          ActivationFunction&& sigma);

    Index GetInputDim() const;
    Index GetOutputDim() const;

    void ZeroGradients();

    Vector ForwardPass(const Vector& x) const;
    Vector BackwardPass(const Vector& x, const Vector& y, const Vector& u,
                        size_t batchSize);

    void Update(std::function<void(Matrix&, const Matrix&)> AUpdater,
                std::function<void(Vector&, const Vector&)> bUpdater);

private:
    Matrix A_;
    Vector b_;
    ActivationFunction sigma_;
    Matrix AGrad_;
    Vector bGrad_;
};

}  // namespace nn
