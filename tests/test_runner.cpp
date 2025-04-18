#include "test_runner.h"

#include "tests.h"

namespace nn {

TestRunner::TestRunner() {
    tests_ = {Test::MNIST()};
}

void TestRunner::RunAllTests() const {
    for (const Test& test : tests_) {
        test.Run();
    }
}

void TestRunner::RunTest(size_t id) const {
    tests_[id].Run();
}

}  // namespace nn
