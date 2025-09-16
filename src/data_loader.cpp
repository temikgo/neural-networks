#include "data_loader.h"

namespace nn {

size_t Batch::size() const {
    return x.cols();
}

DataLoader::Iterator::Iterator(const DataLoader& loader, size_t index)
    : loader_(loader), index_(index) {
}

bool DataLoader::Iterator::operator<(const Iterator& other) const {
    return index_ < other.index_;
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    index_ += loader_.batchSize_;
    return *this;
}

Batch DataLoader::Iterator::operator*() const {
    Batch batch;
    size_t endIndex =
        std::min(index_ + loader_.batchSize_, loader_.features_.size());
    size_t actualBatchSize = endIndex - index_;

    batch.x = Matrix(loader_.features_[0].size(), actualBatchSize);
    batch.y = Matrix(loader_.labels_[0].size(), actualBatchSize);

    for (size_t i = 0; i < actualBatchSize; ++i) {
        size_t idx = loader_.indices_[i + index_];
        batch.x.col(i) = loader_.features_[idx];
        batch.y.col(i) = loader_.labels_[idx];
    }

    return batch;
}

DataLoader::DataLoader(size_t batchSize, unsigned seed)
    : batchSize_(batchSize), rng_(seed) {
}

void DataLoader::Load(const std::filesystem::path& path) {
    Dataset dataset = ReadCSV(path);
    features_ = std::move(dataset.features);
    labels_ = std::move(dataset.labels);

    if (features_.size() != labels_.size()) {
        throw std::runtime_error("Features and labels size mismatch");
    }

    indices_.resize(features_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
}

void DataLoader::Truncate(size_t size) {
    if (size >= features_.size()) {
        return;
    }

    features_.resize(size);
    labels_.resize(size);

    indices_.resize(size);
    std::iota(indices_.begin(), indices_.end(), 0);
}

void DataLoader::Permute() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

void DataLoader::ConvertToOneHot(int min, int max) {
    assert(min <= max && "min cannot be greater than or equal to max");

    std::vector<Vector> oneHotLabels;
    oneHotLabels.reserve(labels_.size());

    for (size_t i = 0; i < labels_.size(); ++i) {
        Scalar labelValue = labels_[i](0);
        int intLabel = static_cast<int>(std::round(labelValue));
        assert(min <= intLabel && intLabel <= max &&
               "Label out of valid class range");
        Vector oneHot = Vector::Zero(max - min + 1);
        oneHot(intLabel - min) = 1.0;
        oneHotLabels.push_back(oneHot);
    }

    labels_ = std::move(oneHotLabels);
}

const std::vector<Vector>& DataLoader::GetFeatures() const {
    return features_;
}

const std::vector<Vector>& DataLoader::GetLabels() const {
    return labels_;
}

Matrix DataLoader::GetFeaturesMatrix() const {
    return VectorToMatrix(features_);
}

Matrix DataLoader::GetLabelsMatrix() const {
    return VectorToMatrix(labels_);
}

DataLoader::Iterator DataLoader::begin() {
    return Iterator(*this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(*this, features_.size());
}

Dataset DataLoader::ReadCSV(const std::filesystem::path& path) {
    static constexpr char delimiter = ',';

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::vector<Vector> features;
    std::vector<Vector> labels;
    std::string line;

    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, delimiter);
        int label = std::stoi(value);
        Vector v(1);
        v(0) = static_cast<Scalar>(label);
        labels.push_back(v);

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

    return {std::move(features), std::move(labels)};
}

Matrix DataLoader::VectorToMatrix(const std::vector<Vector>& vectors) {
    if (vectors.empty()) {
        return Matrix();
    }

    Matrix matrix(vectors[0].size(), vectors.size());
    for (int i = 0; i < vectors.size(); ++i) {
        matrix.col(i) = vectors[i];
    }
    return matrix;
}

}  // namespace nn
