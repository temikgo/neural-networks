#pragma once

#include <functional>

namespace nn {

class Test {
    using RunType = std::function<void()>;

public:
    Test(RunType&& runFunc_);
    void Run() const;

    static Test MNIST();

private:
    RunType runFunc_;
};

}  // namespace nn
