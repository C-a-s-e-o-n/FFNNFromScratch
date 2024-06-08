#pragma once
#include <vector>
#include <iostream>
#include <string>

class Matrix {
public:
	// Default constructor
	Matrix() : rows(0), cols(0) { 
		data.resize(rows, std::vector<double>(cols));
	}

	// Constructor
	Matrix(size_t numRows, size_t numCols, double initVal = 0.0) : rows(numRows), cols(numCols)
	{
		data.resize(rows, std::vector<double>(cols, initVal)); // initialize matrix of vals
	}

	// Accessors
	// used to modify the matrix
	std::vector<double>& operator[](size_t index) {
		return data[index];
	}

	// used to access but not modify, with the same code as the modifying function
	const std::vector<double>& operator[](size_t index) const {
		return data[index];
	}

	size_t numRows() const noexcept {
		return rows;
	}

	size_t numCols() const noexcept {
		return cols;
	}

	void setColumn(size_t colIdx, const std::vector<double>& colData) {
		if (colIdx >= cols) {
			throw std::invalid_argument("Invalid column index.");
		}

		else if (colData.size() != rows) {
			throw std::invalid_argument("Invalid column data size.");
		}

		for (size_t i = 0; i < rows; i++) {
			data[i][colIdx] = colData[i];
		}
	}

	const std::vector<double>& getColumn(size_t colIdx) const {
		if (colIdx >= cols) {
			throw std::out_of_range("Invalid col index.");
		}

		return data[colIdx];
	}

	const std::vector<double>& getRow(size_t rowIdx) const {
		if (rowIdx >= rows) {
			throw std::out_of_range("Invalid row index.");
		}

		return data[rowIdx];
	}

	void shape() const noexcept {
		std::cout << rows << " x " << cols << std::endl;
	}

	std::pair<size_t, size_t> size() const noexcept {
		return {rows, cols};
	}

	void print() const {
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				std::cout << data[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	// operations
	// elementwise addition
	Matrix operator+(const Matrix& other) const {
		if (size() != other.size()) {
			throw std::runtime_error("Matrix dimensions do not match for addition.");
		}

		Matrix result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] + other.data[i][j];
			}
		}
		return result;
	}

	// elementwise subtraction
	Matrix operator-(const Matrix& other) const {
		if (size() != other.size()) {
			throw std::runtime_error("Matrix dimensions do not match for subtraction.");
		}

		Matrix result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] - other.data[i][j];
			}
		}
		return result;
	}

	// scalar addition 
	Matrix operator+(const double scalar) const {
		Matrix result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] + scalar;
			}
		}
		return result;
	}


	// scalar subtraction
	Matrix operator-(const double scalar) const {
		Matrix result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] - scalar;
			}
		}
		return result;
	}

	// matrix multiplication
	Matrix operator*(const Matrix& other) const {
		if (cols != other.numRows()) {
			throw std::runtime_error("Matrix dimensions do not match for multiplication.");
		}

		Matrix result(rows, other.numCols());

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < other.numCols(); j++) {
				double sum = 0.0;
				for (int k = 0; k < cols; k++) {
					sum += data[i][k] * other.data[k][j];
				}
				result.data[i][j] = sum;
			}
		}
		return result;
	}

	// elementwise multiplication
	Matrix elementwiseMult(const Matrix& other) const {
		if (size() != other.size()) {
			this->shape();
			other.shape();
			throw std::runtime_error("Matrix dimensions do not match for elementwise multiplication.");
		}

		Matrix result(rows, cols);

		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] * other.data[i][j];
			}
		}
		return result;
	}

	// scalar multiplication
	Matrix operator*(const double scalar) const {
		Matrix result(rows, cols);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = data[i][j] * scalar;
			}
		}
		return result;
	}

	// transposition
	Matrix T() const {
		Matrix result(cols, rows);

		// reverse indices 
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[j][i] = data[i][j];
			}
		}
		return result;
	}

	// converting from vector to 1-dimensional matrix
	static Matrix toMatrix(const std::vector<int>& vec) noexcept{
		Matrix result(vec.size(), 1);

		for (size_t i = 0; i < vec.size(); i++) {
			result.data[i][0] = vec[i];
		}
		return result;
	}

private:
	size_t rows;
	size_t cols;

	std::vector<std::vector<double>> data;
};