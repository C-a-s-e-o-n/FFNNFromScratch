#include <SFML/graphics.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FFNN.hpp"
#include "MNISTLoader.hpp"
using namespace cv;


int main() {
    try {
        MNISTLoader trainLoader("Data/train-images.idx3-ubyte", "Data/train-labels.idx1-ubyte");
        MNISTLoader testLoader("Data/t10k-images.idx3-ubyte", "Data/t10k-labels.idx1-ubyte");
        //loader.display_images();

        // retrive images as matrix objects
        std::vector<Matrix> mnistTrain = trainLoader.getImages();
        std::vector<int> labelsTrain = trainLoader.getLabels();

        std::vector<Matrix> mnistTest = testLoader.getImages();
        std::vector<int> labelsTest = testLoader.getLabels();

        // print the first image
        /*std::cout << "First Image: " << std::endl;
        mnist_images[0].print();
        std::cout << "First Label: " << std::endl;
        std::cout << labels[0] << std::endl; */

        FFNN model({ 784, 3, 10 }); // 28 * 28 img size = 784 1-D array

        model.SGD(mnistTrain, labelsTrain, 30, 32, 0.01);

        double acc = model.evaluate(mnistTest, labelsTest);

        std::cout << "Overall Model Accuracy: " << acc << std::endl;

    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

	return 0;
}