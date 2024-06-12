#pragma once
#include "Matrix.hpp"
#include "Utils.hpp"

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