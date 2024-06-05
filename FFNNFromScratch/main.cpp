#include <SFML/graphics.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FFNN.hpp"
#include "Matrix.hpp"
using namespace cv;


int main() {
	/*sf::RenderWindow window(sf::VideoMode(800,800), "test");
	sf::Event e;

	while (window.isOpen()) {
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed)
				window.close();
		}
	}*/

	Matrix mat1(2, 1);
	Matrix mat2(1, 2);

	mat1[0][0] = 7.0;
	mat1[1][0] = 3.0;

	Matrix t = mat1.T();
	t.print();



	return 0;
}