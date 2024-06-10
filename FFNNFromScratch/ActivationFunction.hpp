#pragma once
#include "Matrix.hpp"

enum class Activations {
	sigmoid,
	ReLU
};
// Vectorized sigmoid function for matrix input/output
Matrix sigmoid(const Matrix& input) {
    Matrix result(input.numRows(), input.numCols());

    for (size_t i = 0; i < input.numRows(); i++) {
        for (size_t j = 0; j < input.numCols(); j++) {
            double z = input[i][j];
            double sigmoid_z = 1.0 / (1.0 + std::exp(-z)); // Sigmoid function
            result[i][j] = sigmoid_z;
        }
    }

    return result;
}

// Derivative of the sigmoid function
Matrix sigmoidPrime(const Matrix& z) {
    Matrix sig = sigmoid(z);
    Matrix temp = Matrix(sig.numRows(), sig.numCols(), 1.0) - sig;
    return sig.elementwiseMult(temp);
}