#pragma once

#include <Eigen/Dense>

namespace nn {

using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Index = Eigen::Index;

using VectorBatch = std::vector<Vector>;
using MatrixBatch = std::vector<Matrix>;
using NumberBatch = std::vector<int>;

}  // namespace nn
