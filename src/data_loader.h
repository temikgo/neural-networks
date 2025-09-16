#pragma once

#include <filesystem>
#include <fstream>
#include <random>

#include "linalg.h"

namespace nn {

struct Batch {
    Matrix x;
    Matrix y;
    size_t size() const;
};

struct Dataset {
    std::vector<Vector> features;
    std::vector<Vector> labels;
};

class DataLoader {
private:
    class Iterator {
    private:
        const DataLoader& loader_;
        size_t index_;

    public:
        Iterator(const DataLoader& loader, size_t index);

        bool operator<(const Iterator& other) const;

        Iterator& operator++();

        Batch operator*() const;
    };

public:
    DataLoader(size_t batchSize = 128, unsigned seed = 179);

    void Load(const std::filesystem::path& path);

    void Truncate(size_t size);
    void Permute();
    void ConvertToOneHot(int min, int max);

    const std::vector<Vector>& GetFeatures() const;
    const std::vector<Vector>& GetLabels() const;

    Matrix GetFeaturesMatrix() const;
    Matrix GetLabelsMatrix() const;

    Iterator begin();
    Iterator end();

private:
    static Dataset ReadCSV(const std::filesystem::path& path);
    static Matrix VectorToMatrix(const std::vector<Vector>& vectors);

    std::vector<Vector> features_;
    std::vector<Vector> labels_;
    std::vector<size_t> indices_;
    size_t batchSize_;
    std::mt19937 rng_;
};

}  // namespace nn
