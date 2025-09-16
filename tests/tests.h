#pragma once

#include <functional>

namespace nn {

using Test = std::function<void()>;
Test CreateMNISTTest();

}  // namespace nn
