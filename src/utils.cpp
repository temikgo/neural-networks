#include "utils.h"

#include <EigenRand/EigenRand>
#include <unordered_map>

namespace nn {

Matrix GenerateMatrix(Index n, Index m) {
    static Eigen::Rand::Vmt19937_64 urng{179};
    static Eigen::Rand::NormalGen<Scalar> generator{0, 1};
    return generator.generate<Matrix>(n, m, urng);
}

Vector GenerateVector(Index n) {
    return GenerateMatrix(n, 1);
}

Scalar Accuracy(const std::vector<int>& predicted,
                const std::vector<int>& actual) {
    Scalar total = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        total += predicted[i] == actual[i];
    }
    return total / actual.size();
}

Scalar WeightedAccuracy(const std::vector<int>& predicted,
                        const std::vector<int>& actual) {
    std::unordered_map<int, Scalar> weights, correct;
    for (int label : actual) {
        weights[label] += 1;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        correct[actual[i]] += predicted[i] == actual[i];
    }
    Scalar weightedAccuracy = 0;
    for (const auto& [label, weight] : weights) {
        weightedAccuracy +=
            (correct[label] / weight) * (weight / actual.size());
    }
    return weightedAccuracy;
}

Scalar Precision(const std::vector<int>& predicted,
                 const std::vector<int>& actual, int label) {
    Scalar truePositive = 0, falsePositive = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (predicted[i] == label) {
            if (actual[i] == label) {
                ++truePositive;
            } else {
                ++falsePositive;
            }
        }
    }
    return truePositive + falsePositive == 0
               ? 0
               : truePositive / (truePositive + falsePositive);
}

Scalar Recall(const std::vector<int>& predicted, const std::vector<int>& actual,
              int label) {
    Scalar truePositive = 0, falseNegative = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == label) {
            if (predicted[i] == label) {
                ++truePositive;
            } else {
                ++falseNegative;
            }
        }
    }
    return truePositive + falseNegative == 0
               ? 0
               : truePositive / (truePositive + falseNegative);
}

Scalar F1Score(const std::vector<int>& predicted,
               const std::vector<int>& actual, int label) {
    Scalar precision = Precision(predicted, actual, label);
    Scalar recall = Recall(predicted, actual, label);
    return precision + recall == 0
               ? 0
               : 2 * precision * recall / (precision + recall);
}

Matrix ConfusionMatrix(const std::vector<int>& predicted,
                       const std::vector<int>& actual, size_t size) {
    Matrix res = Matrix::Zero(size, size);
    for (size_t i = 0; i < actual.size(); ++i) {
        res(actual[i], predicted[i]) += 1;
    }
    return res;
}

void PrintTrainingHistory(const TrainingHistory& history, std::ostream& out) {
    out << std::fixed << std::setprecision(6);
    for (size_t e = 0; e < history.size(); ++e) {
        const EpochMetrics& em = history[e];
        out << "Epoch " << e << "  "
            << "avg loss     = " << em.AvgLoss() << "  "
            << "avg accuracy = " << em.AvgAccuracy() << "\n";

        out << "  losses: [";
        for (size_t i = 0; i < em.losses.size(); ++i) {
            if (i) {
                out << ", ";
            }
            out << em.losses[i];
        }
        out << "]\n";

        out << "  accuracies: [";
        for (size_t i = 0; i < em.accuracies.size(); ++i) {
            if (i) {
                out << ", ";
            }
            out << em.accuracies[i];
        }
        out << "]\n";
    }
}

bool SaveTrainingHistoryCSV(const TrainingHistory& history,
                            const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        return false;
    }
    f << "epoch,batch,loss,accuracy\n";
    for (size_t e = 0; e < history.size(); ++e) {
        const EpochMetrics& em = history[e];
        for (size_t b = 0; b < em.losses.size(); ++b) {
            f << e << ',' << b << ',' << em.losses[b] << ',' << em.accuracies[b]
              << '\n';
        }
    }
    return true;
}

}  // namespace nn
