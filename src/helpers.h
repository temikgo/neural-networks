#pragma once

#include <vector>

namespace nn {

typedef int Number;

typedef std::vector<Number> Vector;

template <typename T>
using Matrix = std::vector<std::vector<T>>;

}  // namespace nn
