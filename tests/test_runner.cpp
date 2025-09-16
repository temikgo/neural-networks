#include "test_runner.h"

#include <cassert>

#include "tests.h"

namespace nn {

TestRunner::TestRunner() : tests_({CreateMNISTTest()}) {
}

size_t TestRunner::GetTestCount() const {
    return tests_.size();
}

void TestRunner::RunAllTests() const {
    for (const Test& test : tests_) {
        test();
    }
}

void TestRunner::RunTest(size_t id) const {
    assert(0 <= id && id < tests_.size() &&
           "Number of epochs must be positive");
    tests_[id]();
}

}  // namespace nn
