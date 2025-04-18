#include "tests.h"

#include <iostream>
#include <string>

#include "src/activation_function.h"
#include "src/layer.h"
#include "src/linalg.h"
#include "src/loss_function.h"
#include "src/net.h"
#include "src/utils.h"

namespace nn {

Test::Test(RunType&& runFunc) : runFunc_(std::move(runFunc)) {
}

void Test::Run() const {
    runFunc_();
}

static void TestMNIST() {
    const int kMinNumber = 0, kMaxNumber = 9;
    const std::string kTrainPath = "../data/mnist_train.csv";
    const std::string kTestPath = "../data/mnist_test.csv";

    auto [xTrain, labelsTrain] = ReadCSV(kTrainPath, ',');
    auto [xTest, labelsTest] = ReadCSV(kTestPath, ',');

    const int kTruncate = 5000;
    xTrain = VectorBatch(xTrain.begin(), xTrain.begin() + kTruncate);
    labelsTrain = NumberBatch(labelsTrain.begin(), labelsTrain.begin() + kTruncate);
    VectorBatch yTrain = ConvertToOneHot(labelsTrain, kMinNumber, kMaxNumber);
    VectorBatch yTest = ConvertToOneHot(labelsTest, kMinNumber, kMaxNumber);

    std::vector<Index> layerSizes = {xTrain[0].size(), 256, 128, kMaxNumber - kMinNumber + 1};
    std::vector<ActivationFunction> activationFunctions = {
        ActivationFunction::ReLU(), ActivationFunction::ReLU(),
        ActivationFunction::Softmax()};

    std::vector<Layer> layers;
    layers.reserve(activationFunctions.size());
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i - 1], layerSizes[i],
                            std::move(activationFunctions[i - 1]));
    }

    Net net(std::move(layers));
    Optimizer optimizer(1e-4);
    LossFunction loss_function = LossFunction::CrossEntropy();
    net.Train(xTrain, yTrain, 128, 10, loss_function, optimizer);

    VectorBatch logits = net.Predict(xTest);
    NumberBatch predicted;
    predicted.reserve(logits.size());
    for (const Vector& logit : logits) {
        Index argmax;
        Scalar _ = logit.maxCoeff(&argmax);
        predicted.push_back(argmax);
    }

    std::cout << "MNIST accuracy " << Accuracy(predicted, labelsTest) << '\n';
}

Test Test::MNIST() {
    return Test(TestMNIST);
}

}  // namespace nn
