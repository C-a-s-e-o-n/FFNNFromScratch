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
		std::cout << "inputs size: \n";
		inputs.shape();
		std::cout << "weights size: \n";
		weights.shape();
		std::cout << "biases sie: \n";
		biases.shape();

		z = (weights * inputs) + biases; // dont forget order matters with mat mult
		activation_output = sigmoid(z);
	}

	void updateWeights(const Matrix& activation, const Matrix& delta, double learningRate) {

		// update weights
		std::cout << "delta shape\n";
		delta.shape();
		std::cout << "activoat\n";
		activation.T().shape();
		std::cout << "full shape\n";
		(delta.T() * activation * learningRate).shape();
		weights = weights - (delta.T() * activation * learningRate);
		
		// update biases
		biases = biases - (delta * learningRate);
	}

	Matrix getOutput() const {
		return activation_output;
	}
};