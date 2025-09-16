#include "tests.h"

#include <filesystem>
#include <iostream>
#include <string>

#include "src/activation_function.h"
#include "src/builder.h"
#include "src/data_loader.h"
#include "src/layer.h"
#include "src/linalg.h"
#include "src/loss_function.h"
#include "src/net.h"
#include "src/utils.h"

namespace nn {

void PrintMNIST(const std::vector<int>& predicted,
                const std::vector<int>& actual, size_t size) {
    std::cout << "==========================================\n";
    std::cout << "       EVALUATION METRICS FOR MNIST\n";
    std::cout << "==========================================\n\n";

    std::cout << "OVERALL METRICS:\n";
    std::cout << "Accuracy:          " << std::fixed << std::setprecision(4)
              << Accuracy(predicted, actual) << "\n";
    std::cout << "Weighted Accuracy: " << std::fixed << std::setprecision(4)
              << WeightedAccuracy(predicted, actual) << "\n\n";

    std::cout << "PER-CLASS METRICS:\n";
    std::cout << "Class\tPrecision\tRecall\t\tF1-Score\tCount\n";
    std::cout << "-----\t---------\t------\t\t--------\t-------\n";

    std::vector<int> count(size, 0);
    for (int label : actual) {
        ++count[label];
    }

    Scalar macroPrecision = 0, macroRecall = 0, macroF1 = 0;

    for (int label = 0; label < size; ++label) {
        Scalar precision = Precision(predicted, actual, label);
        Scalar recall = Recall(predicted, actual, label);
        Scalar f1 = F1Score(predicted, actual, label);

        macroPrecision += precision;
        macroRecall += recall;
        macroF1 += f1;

        std::cout << label << "\t" << std::fixed << std::setprecision(4)
                  << precision << "\t\t" << std::fixed << std::setprecision(4)
                  << recall << "\t\t" << std::fixed << std::setprecision(4)
                  << f1 << "\t\t" << count[label] << "\n";
    }

    std::cout << "\nMACRO AVERAGE:\n";
    std::cout << "Precision: " << std::fixed << std::setprecision(4)
              << macroPrecision / size << "\n";
    std::cout << "Recall:    " << std::fixed << std::setprecision(4)
              << macroRecall / size << "\n";
    std::cout << "F1-Score:  " << std::fixed << std::setprecision(4)
              << macroF1 / size << "\n\n";

    std::cout << "CONFUSION MATRIX:\n";
    Matrix confusion = ConfusionMatrix(predicted, actual, size);
    std::cout << "Actual \\ Predicted";
    for (int i = 0; i < size; ++i) {
        std::cout << std::setw(5) << i;
    }
    std::cout << "\n";
    for (int i = 0; i < size; ++i) {
        std::cout << std::setw(7) << i << "           ";
        for (int j = 0; j < size; ++j) {
            std::cout << std::setw(5) << static_cast<int>(confusion(i, j));
        }
        std::cout << "\n";
    }

    std::cout << "\n==========================================\n";
}

void TestMNIST() {
    constexpr int kMinNumber = 0, kMaxNumber = 9, kTruncate = 25000;
    const std::filesystem::path kTrainPath = "../data/mnist_train.csv";
    const std::filesystem::path kTestPath = "../data/mnist_test.csv";

    DataLoader trainLoader, testLoader;
    trainLoader.Load(kTrainPath);
    testLoader.Load(kTestPath);

    std::vector<int> originalTestLabels;
    const auto& testLabels = testLoader.GetLabels();
    for (const auto& labelVec : testLabels) {
        originalTestLabels.push_back(static_cast<int>(std::round(labelVec(0))));
    }

    trainLoader.Truncate(kTruncate);
    trainLoader.ConvertToOneHot(kMinNumber, kMaxNumber);
    testLoader.ConvertToOneHot(kMinNumber, kMaxNumber);

    Builder builder;
    Net net = builder.Input(trainLoader.GetFeatures()[0].size())
                  .AddLayer(256, ActivationFunction::ReLU())
                  .AddLayer(128, ActivationFunction::ReLU())
                  .AddLayer(kMaxNumber - kMinNumber + 1,
                            ActivationFunction::Softmax())
                  .Extract();

    Optimizer optimizer(1e-3);
    LossFunction lossFunction = LossFunction::CrossEntropy();
    TrainingHistory history =
        net.Train(trainLoader, 99, lossFunction, optimizer);

    // PrintTrainingHistory(history);
    if (!SaveTrainingHistoryCSV(history, "../output/history.csv")) {
        std::cout << "Save history to CSV failed\n";
    }

    Matrix logits = net.Predict(testLoader.GetFeaturesMatrix());
    std::vector<int> predicted(logits.cols());

    for (int i = 0; i < logits.cols(); ++i) {
        logits.col(i).maxCoeff(&predicted[i]);
    }

    PrintMNIST(predicted, originalTestLabels, kMaxNumber - kMinNumber + 1);
}

Test CreateMNISTTest() {
    return TestMNIST;
}

}  // namespace nn
