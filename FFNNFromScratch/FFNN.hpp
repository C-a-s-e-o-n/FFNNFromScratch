#pragma once
#include "Utils.hpp"
#include <numeric>
#include <random>
#include "Layer.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"

struct Gradients {
    std::vector<Matrix> weightGradients;
    std::vector<Matrix> biasGradients;
};

class FFNN {
public:
    // constructor
    FFNN(const std::vector<int>& layerSizes)
    {
        for (size_t i = 0; i < layerSizes.size() - 1; i++) {
            // ex: layerSizes = {724, 128, 64, 32}
            // layers = (724, 128}, {128, 64}, {64, 32}
            layers.emplace_back(Layer(layerSizes[i + 1], layerSizes[i]));
        }
    }

    // used for testing the model on data after it has been trained
    // inputs = test data
    std::vector<Matrix> forward(const std::vector<Matrix> inputs) {
        std::vector<Matrix> layer_outputs;
        Matrix current_input;



        for (auto& input : inputs) {
            current_input = input.flatten();
            for (auto& layer : layers) {
                layer.feedForward(current_input); // forward pass through the layer
                current_input = layer.getOutput();
            }
            layer_outputs.push_back(current_input);
        }

        return layer_outputs;
    }

    // Stochastic Gradient Descent
    void train(const std::vector<Matrix>& Xtrain, const std::vector<int>& Ytrain, int epochs, int miniBatchSize, double learningRate) {
        assert(Xtrain.size() == Ytrain.size());

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch: " << epoch << "\t";
            double epochLoss = 0.0; // Track error for each epoch

            // Seed the random number generator
            std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());

            // Shuffle training data
            std::vector<size_t> indices(Xtrain.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);

            // Divide data into mini-batches
            for (size_t i = 0; i < Xtrain.size(); i += miniBatchSize) {
                std::vector<Matrix> miniBatchData;
                std::vector<int> miniBatchTargets;

                // Collect mini-batch data and targets
                for (size_t j = i; j < std::min(i + miniBatchSize, Xtrain.size()); j++) {
                    miniBatchData.push_back(Xtrain[indices[j]]);
                    miniBatchTargets.push_back(Ytrain[indices[j]]);
                }

                // Print the mini-batch data and targets
                //std::cout << "Mini-batch data for indices " << i << " to " << std::min(i + miniBatchSize, Xtrain.size()) - 1 << ":" << std::endl;
                //for (size_t k = 0; k < miniBatchData.size(); ++k) {
                    //miniBatchData[k].print();
                    //std::cout << "Label: " << miniBatchTargets[k] << std::endl;
                //}

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

                for (size_t i = 0; i < layers.size(); i++) {
                    layers[i].updateWeightsAndBiases(grad.weightGradients[i], grad.biasGradients[i], learningRate);
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
        // Compute delta for the last layer
        delta[numLayers - 1] = outputs.back() - targets.back();  // Shape should be (numOutputs x 1)

        // Compute weight and bias gradients for the last layer
        weightGradients[numLayers - 1] = delta.back() * layers[numLayers - 2].getOutput().T();  // Shape should be (numOutputs x numNeuronsInPreviousLayer)
        biasGradients[numLayers - 1] = delta.back();  // Shape should be (numOutputs x 1)


        for (int i = numLayers - 2; i >= 0; i--) {
            delta[i] = (layers[i + 1].weights.T() * delta[i + 1]).elementwiseMult(sigmoidPrime(layers[i].z));  // Shape should be (numNeuronsInCurrentLayer x 1)

            if (i > 0) {
                weightGradients[i] = delta[i] * layers[i - 1].getOutput().T();  // Shape should be (numNeuronsInCurrentLayer x numNeuronsInPreviousLayer)
            }
            else {
                weightGradients[i] = delta[i] * inputs[i].flatten().T();  // Shape should be (numNeuronsInCurrentLayer x numInputs)
            }

            biasGradients[i] = delta[i];  // Shape should be (numNeuronsInCurrentLayer x 1)
        }

        return { weightGradients, biasGradients };
    }

    // Evaluation function
    double eval(std::vector<Matrix>& testData, std::vector<int>& targets) {
        int correct = 0;

        // forward pass for the entire test set
        std::vector<Matrix> outputs = forward(testData);

        for (size_t i = 0; i < outputs.size(); i++) {
            int predicted = getPrediction(outputs[i]);
            if (predicted == targets[i]) {
                correct++;
            }
        }

        return (static_cast<double>(correct) / testData.size()) * 100; // Percentage accuracy
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
        return maxIdx;
    }
    int count = 0;
    // one hot encoding
    std::vector<Matrix> createOneHotTargets(const std::vector<int>& targetLabels, int numClasses) {
        count++;
        std::vector<Matrix> oneHotTargets;

        for (const auto& label : targetLabels) {
            Matrix target(numClasses, 1, 0.0); // Initialize the target matrix with zeros
            target[label][0] = 1.0; // set corresponding class label to 1
            oneHotTargets.push_back(target);
        }
        if (count == 1) {
            std::cout << targetLabels[0] << std::endl;
            oneHotTargets[0].print();
        }
        return oneHotTargets;
    }

    // Mean squared error calculation
    double meanSquaredError(const Matrix& predicted, const Matrix& target) {
        assert(predicted.size() == target.size());

        // calc difference between predicted and target values
        Matrix diff = predicted - target;

        // square the differences element-wise
        Matrix squaredDiff = diff.elementwiseMult(diff);

        // calc sum of squared diffs
        double sum = 0.0;
        for (size_t i = 0; i < squaredDiff.numRows(); i++) {
            for (size_t j = 0; j < squaredDiff.numCols(); j++) {
                sum += squaredDiff[i][j];
            }
        }

        // calc mse 
        double mse = sum / (predicted.numRows() * predicted.numCols());

        return mse;
    }

    // Mean squared error derivative
    double meanSquaredErrorDerivative(const Matrix& prediction, int target) {
        return prediction[0][0] - target; // Simplest form for now
    }

private:
    std::vector<Layer> layers; // overall network structure
};