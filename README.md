# OpenCV sudoku solver in QT

Goal: To program a sudoku solver application on QT with OpenCV in C++.

Objectives:

- The program needs to be able to detect the sudoku board with all its contents from an image.
- The program will be passing the structured information of all numbers in the sudoku puzzle to the internal solver.
- The program will get the correct answer, the solved puzzle from the internal solver and lay the answers over the image.

Restrictions: 

- Windows 11 will be the operating system.
- The image is oriented and ready for consumption (computer-generated rather than being taken from a camera)

Approach:

- Isolate the gridlines via thresholding
- Execute edge detection to find to find sudoku gridlines
- Find linear segments via Hough Transform (Skipped)
- Enhance the contours of the gridlines (Skipped)
- Run find contours to locate the largest square
- Calculate the digit-square size 
- Extract / Crop the digit images from the sudoku board
- Train a digit recognition model for C++
	- Train a model for digit recognition in PyTorch
		- Come up with a dataset for digital character recognition
	- Convert to model to ONNX
	- Run ONNX model in C++
- Run digit recognition on the squares
- Get the classified digits together to construct a sudoku board
- Send the board to the internal engine
- Get the response from the internal engine
- Lay the answers on top of the sudoku image
