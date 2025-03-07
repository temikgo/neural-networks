#pragma once

#include <functional>

namespace nn {

class Test {
    using RunType = std::function<void()>;

public:
    Test(RunType&& runFunc_);
    void Run() const;

    static Test TestMSELoss() {
        return Test([]() { int x = 228; });
        // блин мне кажется очень плохо получается так делать тесты, как это
        // правильно делать?
    }

private:
    RunType runFunc_;
};

}  // namespace nn