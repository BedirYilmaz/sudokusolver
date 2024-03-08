#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

Mat im_gray;
int thresh = 100;
const char* source_window = "Source";
const char* contour_window = "Contours";

void create_treshold_image();

int main( int argc, char** argv )
{
    cout << "aa" << endl;
 
    // string im_rgb_path = "C:/Users/3yanl/Code/helloworldcpp/lenna.jpg";
    string im_rgb_path = "C:/Users/3yanl/Code/sudokusolver/sudoku.png";
    Mat im_rgb = imread(im_rgb_path);
    if (!im_rgb.data) {
        return 1;
    }

    im_gray = imread("C:/Users/3yanl/Code/helloworldcpp/sudoku.png", IMREAD_GRAYSCALE);
    cvtColor(im_rgb, im_gray, COLOR_RGB2GRAY);

    blur(im_gray, im_gray, Size(3,3) );

    namedWindow( source_window, WINDOW_NORMAL );
    imshow( source_window, im_gray );

    create_treshold_image();    

    waitKey();
    
    return 0;
}


void create_treshold_image()
{
    Mat canny_output;
    int tresh = 50;
    Canny( im_gray, canny_output, tresh, thresh*2 );
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 255, 0, 0 );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }

    namedWindow( contour_window, WINDOW_NORMAL);
    imshow("Contours", drawing );
}