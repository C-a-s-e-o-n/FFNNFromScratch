#pragma once
#include <numeric>
#include <random>
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

	struct Gradients {
		std::vector<Matrix> weightGradients;
		std::vector<Matrix> biasGradients;
	};

	// used for testing the model on data after it has been trained
	// inputs = test data
	Matrix feedForward(const Matrix& inputs) {
		Matrix activations = inputs;

		for (auto& layer : layers) {
			activations = layer.feedForward(activations);
		}

		return activations;
	}

	void SGD(const std::vector<Matrix>& trainingData, const Matrix& targets, int epochs, int miniBatchSize, double learningRate) {
		for (int epoch = 0; epoch < epochs; epoch++) {
			// shuffle training data
			std::vector<size_t> indices(trainingData.size());
			// generate the range of indices to be shuffled (all), starting from 0
			std::iota(indices.begin(), indices.end(), 0); 
			std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

			// divide data into mini batches
			for (size_t i = 0; i < trainingData.size(); i += miniBatchSize) {
				std::vector<Matrix> miniBatchData;
				Matrix miniBatchTargets(targets.numRows(), 1);

				// collect mini batch data and target
				// for MNIST, trainingData[indices[j]] is a matrix representing 1 full image
				// for MNIST, targets[indices[j]] is the label for that matrix 
				// note that the data has not been shuffled, but rather the indices, changing the order the data is read
				for (size_t j = i; j < std::min(i + miniBatchSize, trainingData.size()); j++) {
					miniBatchData.push_back(trainingData[indices[j]]);
					miniBatchTargets[j] = targets[indices[j]];
				}

				// forward and backward pass for the mini batch
				Gradients grad = backward(miniBatchData, miniBatchTargets, learningRate);

				// update the mini batch weights and biases using the gradients
				for (size_t i = 0; i < layers.size(); i++) {
					layers[i].weights = layers[i].weights - (grad.weightGradients[i] * learningRate);
					layers[i].biases = layers[i].biases - (grad.biasGradients[i] * learningRate);
				}

			}
		}
	}

	// inputs = training data input features, outputs = predictions from forward pass, targets = y_actual
	// input matrix is transpoed to features x samples (features, samples) to simplify matrix operations
	Gradients backward(const std::vector<Matrix>& inputs, Matrix targets, double learningRate) {
		std::vector<Matrix> activations; // vector of activation matrices for each layer
		std::vector<Matrix> zs; // vector of weighted input matrices for each layer

		 // forward pass to store activations and z vectors
		Matrix activation = inputs.back();
		activations.push_back(activation);

		for (size_t i = 0; i < layers.size() - 1; i++) {
			// for each layer, get the weighted inputs from the respective weight matrix & bias vector
			Matrix z = (layers[i].weights * activation) + layers[i].biases;
			zs.push_back(z);
			activation = sigmoid(z);
			activations.push_back(activation);
		}

		// backward pass 
		Matrix lastLayerActivations = activations.back();
		Matrix secondToLastLayerActivations = activations[activations.size() - 2]; // used for initial weight gradients 
		Matrix lastLayerZs = zs.back();

		// error term for last layer of model
		Matrix error = meanSquaredErrorDerivative(lastLayerActivations, targets).elementwiseMult(sigmoidPrime(lastLayerZs));

		// collections of matrices to pass between each layer
		std::vector<Matrix> biasGradients(layers.size()); 
		std::vector<Matrix> weightGradients(layers.size()); 

		// gradients for weights conecting 2nd to last layer to final layer
		weightGradients.back() = error * secondToLastLayerActivations.T();
		// gradients for biases of the last layer neurons
		biasGradients.back() = error;

		for (int i = layers.size() - 2; i >= 0; i--) {
			Matrix z = zs.back();
			Matrix sp = sigmoidPrime(z);

			error = (layers[i + 1].weights.T() * error).elementwiseMult(sp); // transpose to propagate backwards

			biasGradients[i] = error;
			weightGradients[i] = error * activations[i].T();
		}

		// return gradients for update function
		return { weightGradients, biasGradients }; // uses struct definition
	}

	// gives the partial derivative C_x / a , i.e. the sensitivity of the model output towards changes in a
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