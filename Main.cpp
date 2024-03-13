#include <iostream>
#include <string>
#include <cmath> 

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/dnn/dnn.hpp>

#define N 9


using namespace cv;
using namespace dnn;
using namespace std;

Mat im_gray;
Mat im_rgb;
int thresh = 100;
const char* source_window = "Source";
const char* contour_window = "Contours";
const char* answer_window = "Solution";

string modelFilepath{
    "C:/Users/3yanl/Code/sudokusolver/dcr_model.onnx"};
// string modelFilepath{
//     "C:/Users/3yanl/Code/sudokusolver/mnist.onnx"};

dnn::Net net = dnn::readNet(modelFilepath);
bool swapRB = false;
bool crop = false;
void create_treshold_image();


bool isPresentInRow(int grid[N][N], int row, int num) {
    for (int col = 0; col < N; col++) {
        if (grid[row][col] == num) {
            return true;
        }
    }
    return false;
}

bool isPresentInCol(int grid[N][N], int col, int num) {
    for (int row = 0; row < N; row++) {
        if (grid[row][col] == num) {
            return true;
        }
    }
    return false;
}


bool isPresentInBox(int grid[N][N], int startRow, int startCol, int num) {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            if (grid[row + startRow][col + startCol] == num) {
                return true;
            }
        }
    }
    return false;
}

bool isSafe(int grid[N][N], int row, int col, int num) {
    return !isPresentInRow(grid, row, num) &&
           !isPresentInCol(grid, col, num) &&
           !isPresentInBox(grid, row - row % 3, col - col % 3, num);
}

bool findEmptyLocation(int grid[N][N], int &row, int &col) {
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            if (grid[row][col] == 0) {
                return true;
            }
        }
    }
    return false;
}

bool solveSudoku(int grid[N][N]) {
    int row, col;
    if (!findEmptyLocation(grid, row, col)) {
        return true; // Sudoku gelöst
    }
    for (int num = 1; num <= 9; num++) {
        if (isSafe(grid, row, col, num)) {
            grid[row][col] = num;
            if (solveSudoku(grid)) {
                return true;
            }
            grid[row][col] = 0;
        }
    }
    return false;
}

void printGrid(int grid[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            cout << grid[row][col] << " ";
        }
        cout << endl;
    }
}

int main( int argc, char** argv )
{
    cout << "aa" << endl;
 
    // string im_rgb_path = "C:/Users/3yanl/Code/helloworldcpp/lenna.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/b45fa272-de4b-4d66-aca1-0137828efd1e-bestSizeAvailable.jpeg";
    string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Sudoku-Board-1.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Sudoku3.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Sudoku7.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Sudoku11.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Sudoku13.jpg";
    // string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sample_images/Very-easy-difficulty-Sudoku-board-used-for-testing-36-givens.png";

            
    im_rgb = imread(im_rgb_path);
    if (!im_rgb.data) {
        return 1;
    }

    im_gray = imread(im_rgb_path, IMREAD_GRAYSCALE);
    

    create_treshold_image();    
    waitKey();
           


    
    return 0;
}


void create_treshold_image()
{

    cvtColor(im_rgb, im_gray, COLOR_RGB2GRAY);

    blur(im_gray, im_gray, Size(3,3) );

    Mat result;
    int tresh = 50;
    Canny( im_gray, result, tresh, thresh*3, 3 );
        
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( result.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 255, 0, 0 );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }

    // Find rectangles bounding each contour
    vector<Rect> boundingRectangles;
    for (const auto& contour : contours) {
        Rect rect = boundingRect(contour);
        boundingRectangles.push_back(rect);
    }

    // Find the largest rectangle
    Rect largestRect;
    double largestArea = 0;
    for (const auto& rect : boundingRectangles) {
        double area = rect.width * rect.height;
        if (area > largestArea) {
            largestArea = area;
            largestRect = rect;
        }
    }

    

    // find the width and height of a small rectangle
    int s_width = largestRect.width / N;
    int s_height = largestRect.height / N;
    
    int sudokuGrid[N][N] = {}; // This initializes all elements to zero

    // traverse the sudoku boar7d
    vector<Mat> croppedSquares;
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            // char* crop_window = "Crops";
            // namedWindow( crop_window , WINDOW_NORMAL);
            Rect roi((int)(largestRect.x + s_width*i), (int)(largestRect.y + s_height*j), (int)(s_width), (int)(s_height));
            Mat roiImage = im_gray(roi);
            resize(roiImage, roiImage, cv::Size(28, 28));


            croppedSquares.push_back(roiImage);
            // Scalar sum = sum(roiImage)[0];
            cv::Mat notCroppedSquare;
            cv::bitwise_not(roiImage, notCroppedSquare);

            // Define the center region of the ROI by cropping 25% from the edges
            int s_width = notCroppedSquare.cols;
            int s_height = notCroppedSquare.rows;
            cv::Rect roiCenter((int)(s_width * 0.25), (int)(s_height * 0.25), (int)(s_width * 0.5), (int)(s_height * 0.5));

            // Extract the smaller square inside the center region
            cv::Mat smallerSquare = notCroppedSquare(roiCenter).clone();

            // Calculate the sum of all pixel values in the smaller square
            cv::Scalar sumSmallSquare = cv::sum(smallerSquare);
            std::cout << "Sum of all pixels in smaller square " << i << "," << j << " : " << sumSmallSquare << std::endl;

            int inpWidth = roiImage.cols;
            int inpHeight = roiImage.rows;

            Mat blob;
            blobFromImage(roiImage, blob, 1.0, Size(inpWidth, inpHeight), Scalar(), false, false, CV_8U);

            // waitKey();

            net.setInput(blob);
            Mat prob = net.forward();

            // Find the maximum value and its location in the matrix
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(prob, &minVal, &maxVal, &minLoc, &maxLoc);
            
            cout << "All class probs " << prob << endl;
            cout << "Pred index " << maxLoc.x << endl;

            if ((int)(sumSmallSquare[0]) > 1000){
                sudokuGrid[j][i] = maxLoc.x;
            }
            else{
                sudokuGrid[j][i] = 0;
            }

            // imshow(crop_window + to_string(i*9+j), roiImage);
        }
        

    }

    int recognizedSudokuGrid[N][N];

    // Copy the elements from sudokuGrid to recognizedSudokuGrid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            recognizedSudokuGrid[i][j] = sudokuGrid[i][j];
        }
    }

    // Output the initialized grid
    std::cout << "Initialized grid:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << sudokuGrid[i][j] << " ";
        }
        std::cout << std::endl;
    }

    if (solveSudoku(sudokuGrid)) {
        cout << "Sudoku gelöst:\n";
        printGrid(sudokuGrid);
    } else {
        cout << "Keine Lösung gefunden!\n";
    }

    int numberOffset = (int)(s_width / 3);

    // writing the solution on top of the image of the sudoku board
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){

            if (recognizedSudokuGrid[j][i] == 0){
                Point startingPoint((int)(largestRect.x + s_width*i + numberOffset * 0.7) , (int)(largestRect.y + s_height*j + numberOffset*2.5));
                putText(im_rgb, std::to_string(sudokuGrid[j][i]), startingPoint, FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 0, 255), 3);
            }
        }
    }

    // namedWindow(contour_window, WINDOW_NORMAL);
    // imshow(contour_window, drawing );

    namedWindow(answer_window, WINDOW_NORMAL);
    imshow(answer_window, im_rgb );

}