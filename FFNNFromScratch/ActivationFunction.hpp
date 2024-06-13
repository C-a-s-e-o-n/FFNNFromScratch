#pragma once
#include "Matrix.hpp"
#include "Utils.hpp"
#include <cmath>

enum class Activations {
	sigmoid,
	ReLU
};
// Vectorized sigmoid function for matrix input/output
Matrix sigmoid(const Matrix& input) {
    Matrix result(input.numRows(), input.numCols());

    for (size_t i = 0; i < input.numRows(); i++) {
        for (size_t j = 0; j < input.numCols(); j++) {
            result[i][j] = 1.0 / (1.0 + std::exp(-input[i][j])); // Sigmoid function
        }
    }

    return result;
}

// Derivative of the sigmoid function
Matrix sigmoidPrime(const Matrix& z) {
    Matrix sig = sigmoid(z);
    return sig.elementwiseMult(Matrix(sig.numRows(), sig.numCols(), 1.0) - sig);
}

// Vectorized ReLU function for matrix input/output
Matrix relu(const Matrix& input) {
    Matrix result(input.numRows(), input.numCols());

    for (size_t i = 0; i < input.numRows(); i++) {
        for (size_t j = 0; j < input.numCols(); j++) {
            result[i][j] = std::max(0.0, input[i][j]);
        }
    }

    return result;
}

// Derivative of the ReLU function
Matrix reluPrime(const Matrix& z) {
    Matrix result(z.numRows(), z.numCols());

    for (size_t i = 0; i < z.numRows(); i++) {
        for (size_t j = 0; j < z.numCols(); j++) {
            result[i][j] = z[i][j] > 0.0 ? 1.0 : 0.0;
        }
    }

    return result;
}