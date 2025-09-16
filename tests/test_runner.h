#pragma once

#include <vector>

#include "tests.h"

namespace nn {

class TestRunner {
public:
    TestRunner();

    size_t GetTestCount() const;

    void RunAllTests() const;
    void RunTest(size_t id) const;

private:
    std::vector<Test> tests_;
};

}  // namespace nn
