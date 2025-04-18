#include "utils.h"

#include <EigenRand/EigenRand>

#include "src/linalg.h"

namespace nn {

Matrix GenerateMatrix(Index n, Index m) {
    static Eigen::Rand::Vmt19937_64 urng{179};
    static Eigen::Rand::NormalGen<Scalar> generator{0, 1};
    return generator.generate<Matrix>(n, m, urng);
}

Vector GenerateVector(Index n) {
    return GenerateMatrix(n, 1);
}

std::pair<VectorBatch, NumberBatch> ReadCSV(const std::string& path,
                                            char delimiter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    VectorBatch features;
    NumberBatch labels;
    std::string line;

    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, delimiter);
        int label = std::stoi(value);
        labels.push_back(label);

        std::vector<Scalar> pixels;
        while (std::getline(ss, value, delimiter)) {
            pixels.push_back(std::stod(value));
        }

        Vector feature(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
            feature(i) = pixels[i];
        }
        features.push_back(feature);
    }

    return {features, labels};
}

VectorBatch ConvertToOneHot(const NumberBatch& labels, int min, int max) {
    std::vector<Eigen::VectorXd> oneHot;
    oneHot.reserve(labels.size());

    for (int label : labels) {
        Vector vec = Vector::Zero(max - min + 1);
        assert(label >= min && label <= max &&
               "Label out of valid class range");
        vec(label) = 1;
        oneHot.push_back(std::move(vec));
    }

    return oneHot;
}

Scalar Accuracy(const NumberBatch& z, const NumberBatch& y) {
    Scalar total = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        total += z[i] == y[i];
    }
    return total / y.size();
}

}  // namespace nn
