#pragma once
#include "Neuron.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"
#include "Utils.hpp"
#include <random>


class Layer {
public:
	Matrix weights; // all weights for each layer
	Matrix biases; // all biases for each layer
	Matrix z; // weighted sum for each layer;

	Matrix activation_output; // activated z

	Layer(size_t numNeurons, size_t numInputsPerNeuron) :
		weights(numNeurons, numInputsPerNeuron), biases(numNeurons, 1)
	{

		// seed for random number generator
		std::random_device rd;
		std::mt19937 gen(rd());

		// define dist range for random numbers
		// Xavier initialization (NECESSARY FOR RELU)
		std::uniform_real_distribution<> weight_dis(-1.0 / sqrt(numInputsPerNeuron), 1.0 / sqrt(numInputsPerNeuron));
		std::uniform_real_distribution<> bias_dis(0.0, 0.1);

		for (size_t i = 0; i < numNeurons; i++) {
			for (size_t j = 0; j < numInputsPerNeuron; j++) {
				weights[i][j] = weight_dis(gen); 
			}
			biases[i][0] = 0.1;
		}
	}

	void feedForward(const Matrix& inputs) {
		z = (weights * inputs) + biases; // dont forget order matters with mat mult
		activation_output = relu(z);
	}

	void updateWeightsAndBiases(const Matrix& weightGradient, const Matrix& biasGradient, double learningRate, double lambda) {
		// update weights with L2 regularization
		Matrix weightUpdate = weightGradient * learningRate - (weights * lambda);
		weights = weights - weightUpdate;

		// update biases with L2 regularization
		Matrix biasUpdate = biasGradient * learningRate - (biases * lambda);
		biases = biases - biasUpdate;
	}

	Matrix getOutput() const {
		return activation_output;
	}
};