#pragma once
#include <numeric>
#include <random>
#include "Layer.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"
#include "Utils.hpp"

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
	std::vector<Matrix> forward(const std::vector<Matrix> inputs) {
        std::vector<Matrix> layer_outputs;
        Matrix current_input;

		for (auto& input : inputs) {
            current_input = input;
            for (auto& layer : layers) {
                layer.feedForward(current_input); // forward pass through the layer
                current_input = layer.getOutput();
            }
            layer_outputs.push_back(current_input);
		}

		return layer_outputs; 
	}

    // Stochastic Gradient Descent
    void train(const std::vector<Matrix>& Xtrain, const std::vector<int>& Ytrain, int epochs=10, int miniBatchSize=32, double learningRate=0.01) {
        assert(Xtrain.size() == Ytrain.size());

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch: " << epoch << "\t";
            double epochLoss = 0.0; // Track error for each epoch

            // Shuffle training data
            std::vector<size_t> indices(Xtrain.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

            // Divide data into mini-batches
            for (size_t i = 0; i < Xtrain.size(); i += miniBatchSize) {
                std::vector<Matrix> miniBatchData;
                std::vector<int> miniBatchTargets;

                // Collect mini-batch data and targets
                for (size_t j = i; j < std::min(i + miniBatchSize, Xtrain.size()); j++) {
                    miniBatchData.push_back(Xtrain[indices[j]]);
                    miniBatchTargets.push_back(Ytrain[indices[j]]);
                }

                // forward pass for the mini-batch
                std::vector<Matrix> outputs = forward(miniBatchData);

                // encode target vector into a 32 x 1 vector of 10 x 1 matrices
                std::vector<Matrix> oneHotLabels = createOneHotTargets(miniBatchTargets, 10); 

                // calc mse for mini-batch
                double miniBatchLoss = 0.0;
                for (size_t i = 0; i < miniBatchData.size(); i++) {
                    miniBatchLoss += meanSquaredError(outputs[i], oneHotLabels[i]);
                }
                epochLoss += miniBatchLoss;

                // Forward and backward pass for the mini-batch
                Gradients grad = backward(miniBatchData, outputs, oneHotLabels);

                // Update the mini-batch weights and biases using the gradients
                for (size_t i = 0; i < layers.size(); i++) {
                    layers[i].updateWeights(grad.weightGradients[i], grad.biasGradients[i], learningRate);
                }
            }

            // Output epoch loss
            std::cout << "Loss: " << (epochLoss / Xtrain.size()) << std::endl;
        }
    }

    Gradients backward(const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, const std::vector<Matrix>& targets) {
        assert(inputs.size() == outputs.size() && outputs.size() == targets.size());

        size_t numLayers = layers.size();
        std::vector<Matrix> delta(numLayers);
        std::vector<Matrix> weightGradients(numLayers);
        std::vector<Matrix> biasGradients(numLayers);

        // compute delta for last layer
        delta[numLayers - 1] = outputs[numLayers - 1] - targets[numLayers - 1];
        weightGradients[numLayers - 1] = delta[numLayers - 1] * outputs[numLayers - 2].T();
        biasGradients[numLayers - 1] = delta[numLayers - 1];

        // backpropagate error
        for (int i = numLayers - 2; i >= 0; i--) {
            delta[i] = (layers[i + 1].weights.T() * delta[i + 1]).elementwiseMult(sigmoidPrime(layers[i].z));
            weightGradients[i] = delta[i] * inputs[i].T();
            biasGradients[i] = delta[i];
        }

        return { weightGradients, biasGradients };
   }


private:
	std::vector<Layer> layers; // overall network structure
};