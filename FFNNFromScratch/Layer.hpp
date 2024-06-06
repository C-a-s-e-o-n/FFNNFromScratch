#pragma once
#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer {
public:
	Matrix weights; // all weights for each layer
	Matrix biases; // all biases for each layer

	Layer(size_t numNeurons, size_t numInputsPerNeuron) : 
		weights(numNeurons, numInputsPerNeuron), biases(numNeurons, 1)
	{
		for (size_t i = 0; i < numNeurons; i++) {
			for (size_t j = 0; j < numInputsPerNeuron; j++) {
				weights[i][j] = 0.1; // placeholder
			}
			biases[i][0] = 0.1;
		}
	}

	Matrix feedForward(const Matrix& inputs) const {
		return (weights * inputs) + biases;
	}
};