#pragma once
#include "Layer.hpp"

class FFNN {
public:
	// constructor
	FFNN(const std::vector<int>& layerSizes)
	{
		for (size_t i = 1; i < layerSizes.size(); i++) {
			// ex: layerSizes = {724, 128, 64, 32}
			// layers = (724, 128}, {128, 64}, {64, 32}
			layers.emplace_back(Layer(layerSizes[i], layerSizes[i - 1]));
		}
	}

	Matrix feedForward(const Matrix& inputs) {
		Matrix activations(inputs.numRows(), inputs.numCols()); // output vector of previous layer, or initial input of first layer

		for (size_t i = 0; i < inputs.numRows(); i++) {
			activations[i][0] = inputs[i][0];
		}

		for (auto& layer : layers) {
			activations = layer.feedForward(activations);
		}

		return activations;
	}

	void backward(const Matrix& inputs, const Matrix& outputs, const Matrix& targets, double learningRate) {
		std::vector<Matrix> activations; // vector of activation matrices for each layer
		std::vector<Matrix> zs; // vector of weighted input matrices for each layer

		 // forward pass to store activations and z vectors
		Matrix activation = inputs;
		activations.push_back(activation);

		for (size_t i = 0; i < layers.size() - 1; i++) {
			// for each layer, get the weighted inputs from the respective weight matrix & bias vector
			Matrix z = (layers[i].weights * activation) + layers[i].biases;
			zs.push_back(z);
			activation = sigmoid(z);
			activations.push_back(activation);
		}

		// backward pass 
		Matrix& lastLayerActivations = activations.back();
		Matrix& lastLayerZs = zs.back();

		// error term for last layer of model
		Matrix error = meanSquaredErrorDerivative(lastLayerActivations, targets).elementwiseMult(sigmoidPrime(lastLayerZs));

		//ALMOST DONE, NEXT STUFF IS EXTREMELY SIMILAR TO PYTHON CODE, USE DECREASING FOR LOOP
	}

	// gives the partial derivative C_x / a , i.e. the success of the model
	Matrix meanSquaredErrorDerivative(const Matrix& predictions, const Matrix& targets) {
		return predictions - targets;
	}

	// vectorized sigmoid function for matrix input/output
	Matrix sigmoid(Matrix& input) {
		Matrix result(input.numRows(), input.numCols());

		for (size_t i = 0; i < input.numRows(); i++) {
			for (size_t j = 0; j < input.numCols(); j++) {
				double z = input[i][j];
				double sigmoid_z = 1.0 / (1.0 + (std::exp(-z))); // sigmoid function
				result[i][j] = sigmoid_z;
			}
		}

		return result;
	}

	Matrix sigmoidPrime(Matrix& z) {
		Matrix sig = sigmoid(z);
		// 1 - sig, where 1 is a matrix of 1s the same size as sig
		return sig * (Matrix(z.numRows(), z.numCols(), 1.0) - sig);
	}


private:
	std::vector<Layer> layers; // overall network structure
};