#include "test_runner.h"

#include "tests.h"

namespace nn {

TestRunner::TestRunner() { tests_ = {Test::TestMSELoss()}; }

void TestRunner::RunAllTests() const {
    for (const Test& test : tests_) {
        test.Run();
    }
}

void TestRunner::RunTests() const {
    // TODO: add template running
}

}  // namespace nn