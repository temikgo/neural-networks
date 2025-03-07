#include "tests.h"

namespace nn {

Test::Test(RunType&& runFunc) : runFunc_(std::move(runFunc)) {}

void Test::Run() const { runFunc_(); }

}  // namespace nn