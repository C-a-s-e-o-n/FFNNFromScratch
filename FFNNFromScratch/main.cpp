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
        //loader.display_images();

        // retrive images as matrix objects
        std::vector<Matrix> mnistTrain = trainLoader.getImages();
        std::vector<int> labelsTrain = trainLoader.getLabels();

        std::vector<Matrix> mnistTest = testLoader.getImages();
        std::vector<int> labelsTest = testLoader.getLabels();

        // print the first image
        //std::cout << "First Image: " << std::endl;
        // mnistTrain[20].print();
        // std::cout << "First Label: " << std::endl;
        // std::cout << labelsTrain[20] << std::endl; 

        FFNN model({ 784, 128, 64, 10 }); // 28 * 28 img size = 784 1-D array

        int epochs = 20;
        for (int i = 0; i < epochs; i++) {
            model.train(mnistTrain, labelsTrain, 1, 32, 0.01); // Train for one epoch
            double acc = model.eval(mnistTest, labelsTest);
            std::cout << "Epoch: " << i << "\tOverall Model Accuracy: " << acc << std::endl;
        }

        std::string savePath = "Models/ffnn_model.dat";
        std::string loadPath = "Models/ffnn_model.dat";

        // save the model to a file
        saveModel(model.getLayers(), savePath);
        
        // load file into new FFNN object and test loaded model
        FFNN newModel({ 784, 128, 64, 10 });
        loadModel(newModel.getLayers(), loadPath);


        std::cout << "Loaded Model Accuracy: " << newModel.eval(mnistTest, labelsTest) << std::endl;

    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

	return 0;
}