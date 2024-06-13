#pragma once

#include <fstream>
#include <vector>
#include "Layer.hpp"
#include "Matrix.hpp"


void saveModel(const std::vector<Layer>& layers, const std::string& filename) {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file for saving model");
	}

	for (const auto& layer : layers) {
		size_t rows = layer.weights.numRows();
		size_t cols = layer.weights.numCols();

		// save weights
		file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
		file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				file.write(reinterpret_cast<const char*>(&layer.weights[i][j]), sizeof(double));
			}
		}

		// save biases
		rows = layer.biases.numRows();
		cols = layer.biases.numCols();

		file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
		file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				file.write(reinterpret_cast<const char*>(&layer.biases[i][j]), sizeof(double));
			}
		}

		file.close();
	}
}

void loadModel(const std::vector<Layer>& layers, const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file for loading model");
	}

	for (auto& layer : layers) {
		size_t rows, cols;

		// load weights
		file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

		Matrix weights(rows, cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				file.read(reinterpret_cast<char*>(&weights[i][j]), sizeof(double));
			}
		}
		layer.weights = weights;

		// load biases
		file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

		Matrix biases(rows, cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				file.read(reinterpret_cast<char*>(&biases[i][j]), sizeof(double));
			}
		}
		layer.biases = biases;
	}
	file.close();
}