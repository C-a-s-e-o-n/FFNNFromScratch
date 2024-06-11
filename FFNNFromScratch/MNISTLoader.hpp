#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.hpp"
#include "Utils.hpp"

// MNIST is stored in big-endian format, while my system uses little-endian format
uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
	return (val << 16) | (val >> 16);
}

class MNISTLoader {
public:
	MNISTLoader(const std::string& image_filename, const std::string& label_filename) {
		load_mnist(image_filename, label_filename);
	}

	void display_images() {
		char* pixels = new char[rows * cols];

		for (int item_id = 0; item_id < num_items; ++item_id) {
			// read image pixel
			image_file.read(pixels, rows * cols);
			// read label
			label_file.read(&label, 1);

			std::string sLabel = std::to_string(static_cast<int>(label));
			std::cout << "Label is: " << sLabel << std::endl;

			// convert it to cv::Mat and show it
			cv::Mat image_tmp(rows, cols, CV_8UC1, pixels);
			// resize bigger for showing
			cv::resize(image_tmp, image_tmp, cv::Size(200, 200));
			cv::imshow(sLabel, image_tmp);
			cv::waitKey(0);
		}

		delete[] pixels;
	}

	std::vector<Matrix> getImages() const {
		return images;
	}

	std::vector<int> getLabels() const {
		return labels;
	}

private:
	std::ifstream image_file;
	std::ifstream label_file;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;
	char label;

	std::vector<Matrix> images; // store images as custom matrix objects
	std::vector<int> labels; // store labels in vec of ints for FFNN model param

	void load_mnist(const std::string& image_filename, const std::string& label_filename) {
		// Open files
		image_file.open(image_filename, std::ios::in | std::ios::binary);
		label_file.open(label_filename, std::ios::in | std::ios::binary);

		// Read the magic and the meta data
		uint32_t magic;

		image_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2051) {
			throw std::runtime_error("Incorrect image file magic: " + std::to_string(magic));
		}

		label_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2049) {
			throw std::runtime_error("Incorrect label file magic: " + std::to_string(magic));
		}

		image_file.read(reinterpret_cast<char*>(&num_items), 4);
		num_items = swap_endian(num_items);
		label_file.read(reinterpret_cast<char*>(&num_labels), 4);
		num_labels = swap_endian(num_labels);
		if (num_items != num_labels) {
			throw std::runtime_error("Number of images does not match number of labels.");
		}

		image_file.read(reinterpret_cast<char*>(&rows), 4);
		rows = swap_endian(rows);
		image_file.read(reinterpret_cast<char*>(&cols), 4);
		cols = swap_endian(cols);

		std::cout << "Number of images and labels: " << num_items << std::endl;
		std::cout << "Image dimensions: " << rows << "x" << cols << std::endl;

		// Convert cv::Mat to matrix objects for every image
		char* pixels = new char[rows * cols];

		for (int item_id = 0; item_id < num_items; ++item_id) {
			// read image pixel
			image_file.read(pixels, rows * cols);
			// read label
			label_file.read(&label, 1);

			// Convert image data to Matrix object and store
			Matrix img_matrix(rows, cols);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					img_matrix[i][j] = static_cast<double>(pixels[i * cols + j]) / 255.0;
				}
			}
			images.push_back(img_matrix);

			// store label
			labels.push_back(static_cast<int>(label));
		}
		delete[] pixels;
	}
};