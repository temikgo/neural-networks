#pragma once

#include <string>

#include "linalg.h"

namespace nn {

Matrix GenerateMatrix(Index n, Index m);
Vector GenerateVector(Index n);

std::pair<VectorBatch, NumberBatch> ReadCSV(const std::string& path,
                                            char delimiter);

VectorBatch ConvertToOneHot(const NumberBatch& labels, int min, int max);

Scalar Accuracy(const NumberBatch& z, const NumberBatch& y);

}  // namespace nn
