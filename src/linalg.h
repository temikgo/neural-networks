#pragma once

#include <Eigen/Dense>

namespace nn {

using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Index = Eigen::Index;

using RowVector = Eigen::Matrix<Scalar, 1, -1>;

}  // namespace nn
