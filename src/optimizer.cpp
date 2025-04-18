#include "optimizer.h"

namespace nn {

Optimizer::Optimizer(Scalar learningRate) : learningRate_(learningRate) {};

void Optimizer::DoStep(std::vector<Layer>& layers) const {
    const Scalar maxGradNorm = 1;

    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].Update(
            [&](Matrix& A, const Matrix& AGrad) {
                Scalar norm = AGrad.norm();
                if (norm > maxGradNorm) {
                    A -= learningRate_ * (AGrad * (maxGradNorm / norm));
                } else {
                    A -= learningRate_ * AGrad;
                }
            },
            [&](Vector& b, const Vector& bGrad) {
                Scalar norm = bGrad.norm();
                if (norm > maxGradNorm) {
                    b -= learningRate_ * (bGrad * (maxGradNorm / norm));
                } else {
                    b -= learningRate_ * bGrad;
                }
            });
    }
}

}  // namespace nn
