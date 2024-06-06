#include <SFML/graphics.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FFNN.hpp"
using namespace cv;


int main() {
    try {
        // Create a small dataset for testing
        std::vector<Matrix> trainingData;

        Matrix input1(2, 1);
        input1[0][0] = 0.1;
        input1[1][0] = 0.2;
        trainingData.push_back(input1);

        Matrix input2(2, 1);
        input2[0][0] = 0.3;
        input2[1][0] = 0.4;
        trainingData.push_back(input2);

        // Create targets for the training data
        Matrix targets(2, 1);
        targets[0][0] = 0.5;
        targets[1][0] = 0.6;

        std::vector<int> layers = { 2, 3, 1 }; // 2 matrices of data, 3 hidden nodes, 1 output number

        FFNN model(layers);

        model.SGD(trainingData, targets, 100, 1, 0.1);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "An unknown exception occurred" << std::endl;
    }

	return 0;
}