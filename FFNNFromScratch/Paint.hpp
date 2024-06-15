#pragma once
#include <SFML/graphics.hpp>
#include "Matrix.hpp"


class Paint {
public:
	bool mouseDown = false; // detect if mouse is being held & moved 
	Matrix pixelMatrix;


	Paint(sf::RenderWindow& win, FFNN& ffnn) : window(win), model(ffnn)
	{
		pixelMatrix.resize(window.getSize().y, window.getSize().x);
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

				else if (event.type == sf::Event::MouseMoved && mouseDown == 1) {
					userDraw();
				}

				else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
					mouseDown = true;
					userDraw();

					if (clearButton.getGlobalBounds().contains(sf::Vector2f(event.mouseButton.x, event.mouseButton.y))) {
						clearScreen();
					}

					else if (sendButton.getGlobalBounds().contains(sf::Vector2f(event.mouseButton.x, event.mouseButton.y))) {
						//sendToFFNN();
					}
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

			// draw pixels
			sf::Texture texture;
			texture.loadFromImage(image);
			sf::Sprite sprite(texture);
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

		if (mouseY < 400) {
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
		std::vector<Matrix> output = model.forward({ pixelMatrix });
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