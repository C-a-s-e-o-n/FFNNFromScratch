#pragma once
#include "Matrix.hpp"
#include "FFNN.hpp"
#include <cassert>

struct Gradients {
    std::vector<Matrix> weightGradients;
    std::vector<Matrix> biasGradients;
};

// one hot encoding
std::vector<Matrix> createOneHotTargets(const std::vector<int>& targetLabels, int numClasses) {
	std::vector<Matrix> oneHotTargets;

	for (const auto& label : targetLabels) {
		Matrix target(numClasses, 1, 0.0); // Initialize the target matrix with zeros
		target[label][0] = 1.0; // set corresponding class label to 1
		oneHotTargets.push_back(target);
	}
	return oneHotTargets;
}

// Mean squared error calculation
double meanSquaredError(const Matrix& predicted, const Matrix& target) {
    assert(predicted.size() == target.size());

    // calc difference between predicted and target values
    Matrix diff = predicted - target;

    // square the differences element-wise
    Matrix squaredDiff = diff.elementwiseMult(diff);

    // calc sum of squared diffs
    double sum = 0.0;
    for (size_t i = 0; i < squaredDiff.numRows(); i++) {
        for (size_t j = 0; j < squaredDiff.numCols(); j++) {
            sum += squaredDiff[i][j];
        }
    }

    // calc mse 
    double mse = sum / (predicted.numRows() * predicted.numCols());

    return mse;
}

// Mean squared error derivative
double meanSquaredErrorDerivative(const Matrix& prediction, int target) {
    return prediction[0][0] - target; // Simplest form for now
}

/*
// Evaluation function
double evaluate(const std::vector<Matrix>& testData, const std::vector<int>& targets) {
    int correct = 0;

    for (size_t i = 0; i < testData.size(); i++) {
        Matrix output = forward(testData);
        int predicted = getPrediction(output);

        if (predicted == targets[i]) correct++;
    }
    return (static_cast<double>(correct) / testData.size()) * 100; // Percentage accuracy
}

int getPrediction(const Matrix& output) {
    int maxIdx = 0;
    double maxValue = output[0][0];
    for (int i = 1; i < output.numRows(); i++) {
        if (output[i][0] > maxValue) {
            maxValue = output[i][0];
            maxIdx = i;
        }
    }
    return maxIdx + 1;
} */