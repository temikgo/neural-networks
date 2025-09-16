#pragma once

#include <filesystem>
#include <iostream>

#include "linalg.h"
#include "net.h"

namespace nn {

Matrix GenerateMatrix(Index n, Index m);
Vector GenerateVector(Index n);

Scalar Accuracy(const std::vector<int>& predicted,
                const std::vector<int>& actual);
Scalar WeightedAccuracy(const std::vector<int>& predicted,
                        const std::vector<int>& actual);
Scalar Precision(const std::vector<int>& predicted,
                 const std::vector<int>& actual, int label);
Scalar Recall(const std::vector<int>& predicted, const std::vector<int>& actual,
              int label);
Scalar F1Score(const std::vector<int>& predicted,
               const std::vector<int>& actual, int label);
Matrix ConfusionMatrix(const std::vector<int>& predicted,
                       const std::vector<int>& actual, size_t size);

void PrintTrainingHistory(const TrainingHistory& history,
                          std::ostream& out = std::cout);
bool SaveTrainingHistoryCSV(const TrainingHistory& history,
                            const std::filesystem::path& path);

}  // namespace nn
