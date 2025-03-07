#pragma once

#include <vector>

#include "tests.h"

namespace nn {

class TestRunner {
public:
    TestRunner();

    void RunAllTests() const;
    void RunTests() const;

private:
    std::vector<Test> tests_;
};

}  // namespace nn