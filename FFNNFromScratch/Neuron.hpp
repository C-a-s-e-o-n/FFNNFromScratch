#pragma once

class Neuron {
public:
	Neuron(size_t numInputs) : numInputs(numInputs)
	{
		// initialize weights/bias randomly
	}

	double feedForward(const std::vector<double>& inputs) {
		if (inputs.size() != numInputs) {
			throw std::runtime_error("Input size does not match neuron's expected amount of inputs");
		}

		double sum = 0.0;
		for (size_t i = 0; i < numInputs; i++) {
			sum += inputs[i] * weights[i];
		}
		sum += bias;

		return sigmoid(sum);
	}

	double sigmoid(double x) {
		return 1 / 1 + std::exp(-x);
	}

	void setWeights(const std::vector<double>& weights) {
		if (weights.size() != numInputs) {
			throw std::runtime_error("Weights size does not match input size for neuron");
		}

		this->weights = weights;
	}

	void setBias(double bias) {
		this->bias = bias;
	}

	double getBias() const {
		return bias;
	}

private:
	size_t numInputs;

	std::vector<double> weights;
	double bias;
};