#pragma once
#include <SFML/graphics.hpp>
#include "Matrix.hpp"


class Paint {
public:
	bool mouseDown = false; // detect if mouse is being held & moved 
	Matrix pixelMatrix;


	Paint(sf::RenderWindow& win) : window(win)
	{
		pixelMatrix.resize(window.getSize().y, window.getSize().x);

		run();
	}

	void run() {
		while (window.isOpen()) {
			sf::Event event;

			while (window.pollEvent(event)) {
				if (event.type == sf::Event::Closed) {
					window.close();
				}

				else if (event.type == sf::Event::MouseMoved && mouseDown == 1) {
					userDraw();
				}

				else if (event.type == sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					mouseDown = true;
					userDraw();
				}

				else if (event.type == sf::Event::MouseButtonReleased) {
					mouseDown = false;
				}
			}

			window.clear(sf::Color::White);

			sf::Image image;
			image.create(pixelMatrix.numCols(), pixelMatrix.numRows(), sf::Color::White);

			for (size_t i = 0; i < pixelMatrix.numRows(); i++) {
				for (size_t j = 0; j < pixelMatrix.numCols(); j++) {
					int grey = static_cast<int>(255 * (1.0 - pixelMatrix[i][j])); // convert pixel val to int for RGB val
					image.setPixel(j, i, sf::Color(grey, grey, grey));
				}
			}

			sf::Texture texture;
			texture.loadFromImage(image);
			sf::Sprite sprite(texture);
			window.draw(sprite);

			window.display();
		}
	}

	void userDraw() {
		sf::Vector2i mousePos = sf::Mouse::getPosition(window);
		int mouseX = mousePos.x;
		int mouseY = mousePos.y;

		// blurring effect to simulate MNIST image data
		setPixel(mouseY, mouseX, 1.0); // Center pixel
		setPixel(mouseY - 1, mouseX, 0.5); // Top pixel
		setPixel(mouseY + 1, mouseX, 0.5); // Bottom pixel
		setPixel(mouseY, mouseX - 1, 0.5); // Left pixel
		setPixel(mouseY, mouseX + 1, 0.5); // Right pixel
		setPixel(mouseY - 1, mouseX - 1, 0.25); // Top-left pixel
		setPixel(mouseY - 1, mouseX + 1, 0.25); // Top-right pixel
		setPixel(mouseY + 1, mouseX - 1, 0.25); // Bottom-left pixel
		setPixel(mouseY + 1, mouseX + 1, 0.25); // Bottom-right pixel
												

	}

	// set pixel values manually
	void setPixel(size_t row, size_t col, double val) {
		if (row >=  0 && col >= 0 && row < static_cast<int>(pixelMatrix.numRows())
			&& col < static_cast<int>(pixelMatrix.numCols())) {
			pixelMatrix[row][col] = val;
		}
	}







private:
	sf::RenderWindow& window;
	sf::RenderTexture renderTexture;
	sf::Texture texture;
	sf::Sprite sprite;
};