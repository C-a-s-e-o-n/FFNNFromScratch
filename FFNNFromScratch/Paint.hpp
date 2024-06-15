#pragma once
#include <SFML/graphics.hpp>
#include "Matrix.hpp"
#include <algorithm>


class Paint {
public:
	bool mouseDown = false; // detect if mouse is being held & moved 
	Matrix pixelMatrix;


	Paint(sf::RenderWindow& win, FFNN& ffnn) : window(win), model(ffnn)
	{
		pixelMatrix.resize(100, 100);
		createButtons();
		run();
	}

	void run() {
		while (window.isOpen()) {
			sf::Event event;

			while (window.pollEvent(event)) {
				if (event.type == sf::Event::Closed) {
					window.close();
				}

				if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
					mouseDown = true;

					// Check if the clear button is clicked
					if (clearButton.getGlobalBounds().contains(sf::Vector2f(event.mouseButton.x, event.mouseButton.y))) {
						clearScreen();
					}
					// Check if the send button is clicked
					else if (sendButton.getGlobalBounds().contains(sf::Vector2f(event.mouseButton.x, event.mouseButton.y))) {
						sendToFFNN();
					}
				}

				if (event.type == sf::Event::MouseMoved && mouseDown == 1) {
					userDraw();
				}

				else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
					mouseDown = true;
					userDraw();
				}

				else if (event.type == sf::Event::MouseButtonReleased) {
					mouseDown = false;
				}
			}

			window.clear(sf::Color::White);

			// Draw the drawing area border
			sf::RectangleShape border(sf::Vector2f(100, 100));
			border.setFillColor(sf::Color::Transparent);
			border.setOutlineColor(sf::Color::Black);
			border.setOutlineThickness(2);

			// Calculate the position to center horizontally
			float borderX = (window.getSize().x - 100) / 2.0f;
			float borderY = 100;

			border.setPosition(borderX, 100);

			window.draw(border);
			
			sf::Image image;
			image.create(pixelMatrix.numCols(), pixelMatrix.numRows(), sf::Color::White);

			for (size_t i = 0; i < pixelMatrix.numRows(); i++) {
				for (size_t j = 0; j < pixelMatrix.numCols(); j++) {
					int grey = static_cast<int>(255 * (1.0 - pixelMatrix[i][j])); // convert pixel val to int for RGB val
					image.setPixel(j, i, sf::Color(grey, grey, grey));
				}
			}

			// draw pixels
			sf::Texture texture;
			texture.loadFromImage(image);
			sf::Sprite sprite(texture);
			sprite.setPosition(borderX, 100); // Center horizontally


			window.draw(sprite);

			// draw buttons & text
			window.draw(clearButton);
			window.draw(sendButton);

			window.draw(clearButtonText);
			window.draw(sendButtonText);

			window.display();
		}
	}

	void createButtons() {
		// button for clearing screen
		clearButton.setSize(sf::Vector2f(200, 50));
		clearButton.setFillColor(sf::Color::Black);
		clearButton.setPosition(20, 430);

		// button for sending data to ffnn
		sendButton.setSize(sf::Vector2f(200, 50));
		sendButton.setFillColor(sf::Color::Black);
		sendButton.setPosition(280, 430);

		if (!font.loadFromFile("Fonts/ARIAL.TTF")) {
			std::cerr << "Failed to load font" << std::endl;
		}

		// text on clear button
		clearButtonText.setFont(font);
		clearButtonText.setString("Clear");
		clearButtonText.setCharacterSize(30);
		clearButtonText.setFillColor(sf::Color::White);
		clearButtonText.setPosition(clearButton.getPosition().x + 60, clearButton.getPosition().y + 3);

		// text on send button
		sendButtonText.setFont(font);
		sendButtonText.setString("Send");
		sendButtonText.setCharacterSize(30);
		sendButtonText.setFillColor(sf::Color::White);
		sendButtonText.setPosition(sendButton.getPosition().x + 70, sendButton.getPosition().y + 3);
	}

	void userDraw() {
		sf::Vector2i mousePos = sf::Mouse::getPosition(window);
		int mouseX = mousePos.x;
		int mouseY = mousePos.y;

		// Calculate the position to center horizontally
		float borderX = (window.getSize().x - 100) / 2.0f;
		float borderY = 100;

		if (mouseY >= borderY && mouseY < borderY + 100) {
			// Convert the mouse position relative to the drawing area
			int adjustedMouseX = mouseX - borderX;
			int adjustedMouseY = mouseY - borderY;

			// Ensure that the adjusted mouse position is within the drawing area
			if (adjustedMouseX >= 0 && adjustedMouseX < 100 && adjustedMouseY >= 0 && adjustedMouseY < 100) {
				// blurring effect to simulate MNIST image data
				setPixel(adjustedMouseY, adjustedMouseX, 1.0); // Center pixel
				setPixel(adjustedMouseY - 1, adjustedMouseX, 0.5); // Top pixel
				setPixel(adjustedMouseY + 1, adjustedMouseX, 0.5); // Bottom pixel
				setPixel(adjustedMouseY, adjustedMouseX - 1, 0.5); // Left pixel
				setPixel(adjustedMouseY, adjustedMouseX + 1, 0.5); // Right pixel
				setPixel(adjustedMouseY - 1, adjustedMouseX - 1, 0.25); // Top-left pixel
				setPixel(adjustedMouseY - 1, adjustedMouseX + 1, 0.25); // Top-right pixel
				setPixel(adjustedMouseY + 1, adjustedMouseX - 1, 0.25); // Bottom-left pixel
				setPixel(adjustedMouseY + 1, adjustedMouseX + 1, 0.25); // Bottom-right pixel
			}
		}
	}

	// set pixel values manually
	void setPixel(size_t row, size_t col, double val) {
		if (row >=  0 && col >= 0 && row < static_cast<int>(pixelMatrix.numRows())
			&& col < static_cast<int>(pixelMatrix.numCols())) {
			pixelMatrix[row][col] = val;
		}
	}

	void clearScreen() {
		for (size_t i = 0; i < pixelMatrix.numRows(); i++) {
			for (size_t j = 0; j < pixelMatrix.numCols(); j++) {
				pixelMatrix[i][j] = 0.0;
			}
		}
	}

	void sendToFFNN() {
		Matrix temp = pixelMatrix;
		std::cout << "TEP INITIAL\n";
		temp.print();
		std::cout << "TEP RESIZED\n";

		temp.resize(28, 28);
		// Copy the content of the pixelMatrix to the center of the temp matrix
		for (size_t i = 0; i < std::min(pixelMatrix.numRows(), static_cast<size_t>(100)); ++i) {
			for (size_t j = 0; j < std::min(pixelMatrix.numCols(), static_cast<size_t>(100)); ++j) {
				temp[i + (28 - pixelMatrix.numRows()) / 2][j + (28 - pixelMatrix.numCols()) / 2] = pixelMatrix[i][j];
			}
		}
		temp.print();
		temp.flatten();
		std::cout << "TEP RESIZED AND FLATTENED\n";

		temp.print();
		std::vector<Matrix> output = model.forward({ temp });
		std::cout << model.getPrediction(output[0]);
	}

private:
	sf::RenderWindow& window;
	sf::RenderTexture renderTexture;
	sf::Texture texture;
	sf::Sprite sprite;

	sf::Text clearButtonText;
	sf::Text sendButtonText;

	sf::Font font;

	sf::RectangleShape clearButton;
	sf::RectangleShape sendButton;

	FFNN& model;
};