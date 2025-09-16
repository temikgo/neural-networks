#pragma once

#include "activation_function.h"
#include "linalg.h"

namespace nn {

enum In : Index;
enum Out : Index;
class Layer {
public:
    Layer(In inputDim, Out outputDim,
          ActivationFunction&& sigma = ActivationFunction::Id());
    Layer(Matrix&& A, Vector&& b,
          ActivationFunction&& sigma = ActivationFunction::Id());

    Index GetInputDim() const;
    Index GetOutputDim() const;

    void ZeroGradients();

    Matrix ForwardPass(const Matrix& x) const;
    Matrix BackwardPass(const Matrix& xBatch, const Matrix& yBatch,
                        const Matrix& uBatch);

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
