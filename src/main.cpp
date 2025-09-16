#include "except.h"
#include "tests/test_runner.h"

int main() {
    try {
        nn::TestRunner runner;
        runner.RunAllTests();
    } catch (...) {
        except::react();
    }
    return 0;
}
