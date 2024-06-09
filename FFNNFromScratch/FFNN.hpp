#pragma once
#include <numeric>
#include <random>
#include "Layer.hpp"
#include "Matrix.hpp"

class FFNN {
public:
	// constructor
	FFNN(const std::vector<int>& layerSizes)
	{
		for (size_t i = 0; i < layerSizes.size()-1; i++) {
			// ex: layerSizes = {724, 128, 64, 32}
			// layers = (724, 128}, {128, 64}, {64, 32}
			layers.emplace_back(Layer(layerSizes[i+1], layerSizes[i]));
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

	void SGD(const std::vector<Matrix>& trainingData, const std::vector<int>& targets, int epochs, int miniBatchSize, double learningRate) {
		double loss = 0.0; // track total loss across epochs
		for (int epoch = 0; epoch < epochs; epoch++) {
			std::cout << "Epoch: " << epoch << "\t";
			double epochLoss = 0.0; // track loss for each epoch

			// shuffle training data
			std::vector<size_t> indices(trainingData.size());
			// generate the range of indices to be shuffled (all), starting from 0
			std::iota(indices.begin(), indices.end(), 0); 
			std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

			// divide data into mini batches
			for (size_t i = 0; i < trainingData.size(); i += miniBatchSize) {
				std::vector<Matrix> miniBatchData;
				std::vector<int> miniBatchTargets;

				// collect mini batch data and target
				// for MNIST, trainingData[indices[j]] is a matrix representing 1 full image
				// for MNIST, targets[indices[j]] is the label for that matrix 
				// note that the data has not been shuffled, but rather the indices, changing the order the data is read
				for (size_t j = i; j < std::min(i + miniBatchSize, trainingData.size()); j++) {
					miniBatchData.push_back(trainingData[indices[j]]);
					miniBatchTargets.push_back(targets[indices[j]]);
				}

				// forward and backward pass for the mini batch
				Gradients grad = backward(miniBatchData, miniBatchTargets);

				// update the mini batch weights and biases using the gradients
				for (size_t i = 0; i < layers.size(); i++) {
					layers[i].weights = layers[i].weights - (grad.weightGradients[i] * learningRate);
					layers[i].biases = layers[i].biases - (grad.biasGradients[i] * learningRate);
				}

				double miniBatchLoss = meanSquaredError(miniBatchData, miniBatchTargets);
				epochLoss += miniBatchLoss;
			}
			// Output epoch loss
			std::cout << "Loss: " << epochLoss << std::endl;
		}
	}

	// inputs = training data input features, outputs = predictions from forward pass, targets = y_actual
	// input matrix is transpoed to features x samples (features, samples) to simplify matrix operations
	Gradients backward(const std::vector<Matrix>& inputs, std::vector<int>& targets) {
		std::vector<Matrix> activations; // vector of activation matrices for each layer
		std::vector<Matrix> zs; // vector of weighted input matrices for each layer
		std::vector<Matrix> lastLayerZs; 
		std::vector<Matrix> secondToLastLayerActivations; // used for initial weight gradients 
		Matrix lastLayerActivations(inputs.size(), 1);



		 // forward pass to store activations and z vectors

		for (int i = 0; i < inputs.size(); i++) {
			Matrix activation = inputs[i].flatten(); // flatten to 1D vectors

			std::vector<Matrix> batch_activations; // for storing activations of each layer for current input
			for (int j = 0; j < layers.size(); j++) {
				// for each layer, get the weighted inputs from the respective weight matrix & bias vector
				layers[j].weights.shape();
				layers[j].biases.shape();

				Matrix z = (layers[j].weights * activation) + layers[j].biases;
				zs.push_back(z);
				activation = sigmoid(z);
				batch_activations.push_back(activation);
			}
			activations.push_back(batch_activations.back()); // store only the final activation for each image
			lastLayerZs.push_back(zs.back().getColumn(zs[i].numCols()-1)); // store the final weighted inputs of each layer
			// store second to last layer activations for beginning the inital 
			secondToLastLayerActivations.push_back(batch_activations[batch_activations.size() - 2]); 
		}
		
		// backward pass 

		// error term for last layer of model
		std::vector<Matrix> error(inputs.size(), Matrix(10,1));
		for (int i = 0; i < inputs.size(); i++) {
			double prediction = getPrediction(activations[i]);
			int target = targets[i];
			for (int j = 0; j < 10; j++) {
				Matrix sample()
				double sampleError = meanSquaredErrorDerivative(prediction, target);
				error[i][j][0] = sampleError;
			}
		}

		// Compute error for the last layer for each sample
		for (int i = 0; i < inputs.size(); i++) {
			error[i] = error[i].elementwiseMult(sigmoidPrime(lastLayerZs[i]));
		}

		// collections of matrices to pass between each layer
		std::vector<Matrix> biasGradients(layers.size()); 
		std::vector<Matrix> weightGradients(layers.size()); 

		for (int i = 0; i < inputs.size(); i++) {
			weightGradients[i] = error[i] * secondToLastLayerActivations[i].T();
			biasGradients[i] = error[i];
		}

		for (int i = 0; i < inputs.size(); i++) {
			Matrix z = zs[i];
			Matrix sp = sigmoidPrime(z);

			Matrix delta = (layers.back().weights.T() * error[i]).elementwiseMult(sp); // transpose to propagate backwards

			biasGradients.back() = error[i];
			weightGradients.back() = error[i] * secondToLastLayerActivations[i].T();

			for (int j = layers.size() - 2; j >= 0; j--) {
				error[i] = (layers[j].weights.T() * delta).elementwiseMult(sigmoidPrime(zs[j]));
				biasGradients[j] = error[i];
				weightGradients[j] = error[i] * activations[i].T();
				delta = error[i];
				z = zs[j];
			}
		}
		
		// return gradients for update function
		return { weightGradients, biasGradients }; // uses struct definition
	}

	double meanSquaredError(const std::vector<Matrix>& inputs, const std::vector<int>& targets) {
		double loss = 0.0;
		for (size_t i = 0; i < inputs.size(); ++i) {
			Matrix predictions = feedForward(inputs[i]);
			Matrix actuals = Matrix::toMatrix({ targets[i] });
			Matrix diff = predictions - actuals;
			double sum = 0.0;
			for (size_t j = 0; j < diff.numRows(); ++j) {
				for (size_t k = 0; k < diff.numCols(); ++k) {
					sum += diff[j][k] * diff[j][k];
				}
			}
			loss += sum / (diff.numRows() * diff.numCols());
		}
		return loss / inputs.size();
	}

	// gives the partial derivative C_x / a , i.e. the sensitivity of the model output towards changes in a
	double meanSquaredErrorDerivative(const double prediction, const int target) {
		return prediction - target;
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
		Matrix temp = Matrix(sig.numRows(), sig.numCols(), 1.0) - sig;
		return sig.elementwiseMult(temp);
	}

	double evaluate(const std::vector<Matrix>& testData, const std::vector<int>& targets) {
		int correct = 0;

		for (size_t i = 0; i < testData.size(); i++) {
			Matrix output = feedForward(testData[i]);
			int predicted = getPrediction(output);

			if (predicted == targets[i]) correct++; 
		}
		return (static_cast<double>(correct) / testData.size()) * 100; // percentage acc
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
		return maxIdx+1;
	}


private:
	std::vector<Layer> layers; // overall network structure
};