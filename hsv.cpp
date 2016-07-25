#include<cv.h>
#include<iostream>
#include<highgui.h>

using namespace std;
using namespace cv;

int  main()
{
	Mat mat = imread("./test_images/face.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat dst;
	cvtColor(mat, dst, CV_RGB2HSV);
	vector<Mat> vec;
	vec.reserve(3);
	split(dst, vec);
	imshow("img1", vec[0]);
	imshow("img2", vec[1]);
	imshow("img3", vec[2]);
	waitKey(0);
}
