#pragma once
#include "Neuron.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"

class Layer {
public:
	Matrix weights; // all weights for each layer
	Matrix biases; // all biases for each layer
	Matrix z; // weighted sum for each layer;

	Matrix activation_output; // activated z

	Layer(size_t numNeurons, size_t numInputsPerNeuron) : 
		weights(numNeurons, numInputsPerNeuron), biases(numNeurons, 1)
	{
		for (size_t i = 0; i < numNeurons; i++) {
			for (size_t j = 0; j < numInputsPerNeuron; j++) {
				weights[i][j] = 0.1; 
			}
			biases[i][0] = 0.1;
		}
	}

	void feedForward(const Matrix& inputs) {
		z = (weights * inputs) + biases; // dont forget order matters with mat mult
		activation_output = sigmoid(z);
	}

	void updateWeightsAndBiases(const Matrix& weightGradient, const Matrix& biasGradient, double learningRate) {
		// update weights and biases
		weights = weights - (weightGradient * learningRate);
		biases = biases - (biasGradient * learningRate);
	}

	Matrix getOutput() const {
		return activation_output;
	}
};