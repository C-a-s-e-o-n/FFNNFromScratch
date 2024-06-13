#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
#include <SFML/graphics.hpp>
#include <iostream>
#include "Utils.hpp"
#include "FFNN.hpp"
#include "MNISTLoader.hpp"
#include "Serialize.hpp"



int main() {
    try {
        MNISTLoader trainLoader("Data/train-images.idx3-ubyte", "Data/train-labels.idx1-ubyte");
        MNISTLoader testLoader("Data/t10k-images.idx3-ubyte", "Data/t10k-labels.idx1-ubyte");

        // retrieve images as matrix objects
        std::vector<Matrix> mnistTrain = trainLoader.getImages();
        std::vector<int> labelsTrain = trainLoader.getLabels();

        std::vector<Matrix> mnistTest = testLoader.getImages();
        std::vector<int> labelsTest = testLoader.getLabels();

        FFNN model({ 784, 128, 64, 10 }); // 28 * 28 img size = 784 1-D array

        std::string loadPath = "Models/ffnn_model.dat";
        
        // load file into new FFNN object and test loaded model
        loadModel(model.getLayers(), loadPath);

        // TODO: get the input image from the user, with an SFML drawing app that allows digits to be manually drawn
        // get the digits, normalize the values, and resize the vector into a 28 * 28 and then flatten and forward pass


        // test a specific training example using the model: 
        int exIdx = 400;
        Matrix singleTestImage = mnistTrain[exIdx];
        int label = labelsTrain[exIdx];

        std::vector<Matrix> inputVec = { singleTestImage };
        std::vector<Matrix> outputVec = model.forward(inputVec);
        Matrix output = outputVec[0];

        // get predicted label
        int predictedLabel = model.getPrediction(output);

        std::cout << "Actual label: " << label << std::endl;
        std::cout << "Predicted label: " << predictedLabel << std::endl;


    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

	return 0;
}