#include<cv.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>

using namespace std;
using namespace cv;

/*
@brief：获取并写入图片RGB三个通道的像素值
*/
void getBGRpixels(IplImage* srcImg)
{
	//获取彩色图像的各通道像素值信息
	int height = srcImg->height;
	int width = srcImg->width;
	ofstream infile1,infile2,infile3;
	infile1.open("./test_res/b_img.txt");
	infile2.open("./test_res/g_img.txt");
	infile3.open("./test_res/r_img.txt");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int temp1 = cvGet2D(srcImg, i, j).val[0];
			infile1 << temp1 << " ";
			int temp2 = cvGet2D(srcImg, i, j).val[1];
			infile2 << temp2 << " ";
			int temp3 = cvGet2D(srcImg, i, j).val[2];
			infile3 << temp3 << " ";
		}
		infile1 << endl;
		infile2 << endl;
		infile3 << endl;
	}
	infile1.close();
	infile2.close();
	infile3.close();
}

/*
@brief：读取并写入灰度图的每个像素值
*/
void getGrayPixels(IplImage* srcImg)
{
	int height = srcImg->height;
	int width = srcImg->width;
	ofstream infile;
	infile.open("./test_res/cropped_face2.txt",ios::app);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int temp = cvGet2D(srcImg, i, j).val[0];//读取每个像素值
			//cout << temp << endl;
			infile << temp << " ";
		}
		infile << endl;
	}
	infile.close();
}

/*
@brief：将原图片分割成左上右上中心左下右下感兴趣的五部分，并保存在容器中
@return：元素为pair的容器
*/
vector<pair<int, IplImage*>> segementImage(IplImage* srcImg)
{
	int width = srcImg->width / 3;
	int height = srcImg->height / 3;

	vector<CvRect> vec;
	vector<CvRect>::iterator it;
	vector<pair<int, IplImage*>> pHist;
	int parts;
	CvRect rect;
	for (int i = 1; i < 6; i++)
	{
		switch (i)
		{
		case 1:
			rect.x = 0, rect.y = 0, rect.width = width, rect.height = height;
			vec.push_back(rect);
			break;
		case 2:
			rect.x = 2 * width, rect.y = 0, rect.width = width, rect.height = height;
			vec.push_back(rect);
			break;
		case 3:
			rect.x = width, rect.y = height, rect.width = width, rect.height = height;
			vec.push_back(rect);
			break;
		case 4:
			rect.x = 0, rect.y = 2 * height, rect.width = width, rect.height = height;
			vec.push_back(rect);
			break;
		case 5:
			rect.x = 2 * width, rect.y = 2 * height, rect.width = width, rect.height = height;
			vec.push_back(rect);
			break;
		default:
			break;
		}
	}
	/*for (it = vec.begin(); it != vec.end(); it++)
	{
		cout << (*it).x << " ";
	}*/
	int count = 1;
	for (it = vec.begin(); it != vec.end(); it++)
	{
		IplImage* dst = cvCreateImage(cvSize(width, height), srcImg->depth, 1);
		cvSetImageROI(srcImg, *it);
		cvCopy(srcImg, dst);
		pHist.push_back(make_pair(count, dst));
		cvResetImageROI(srcImg);
		count++;
	}
	return pHist;
}

/*
@brief：计算灰度图的直方图
@return：histogram
*/
CvHistogram* calcGrayHist(IplImage **srcImage)
{
	int HistSize = 16;
	float ranges[] = { 0, 255 };//灰度级范围
	float *pranges[] = { ranges };
	CvHistogram* histogram = cvCreateHist(1, &HistSize, CV_HIST_ARRAY, pranges, 1);
	cvCalcHist(srcImage, histogram);
	return histogram;
}

/*
@brief：提取灰度图四角以及中心的直方图像素
path：写入像素值信息的路径
*/
void extractCornersAndCenter(IplImage* srcImg, CvHistogram* srcHist,string path)
{
	int bins = 16;//直方图要归到几个bin里面
	float bin_val;//统计得到每个bin里面的像素个数
	vector<float> vec;
	vector<float>::iterator it;
	for (int i = 0; i < bins; i++)
	{
		float bin_val = cvQueryHistValue_1D(srcHist, i);
		vec.push_back(bin_val);
	}
	//取最大最小值
	//float max_bin = *max_element(vec.begin(), vec.end());
	//float min_bin = *min_element(vec.begin(), vec.end());

	////归一化
	//for (it = vec.begin(); it != vec.end(); it++)
	//{
	//	*(it) = (*(it)-min_bin) / (max_bin - min_bin);
	//}

	//写入文件
	ofstream infile;
	infile.open(path,ios::app);
	for (it = vec.begin(); it != vec.end(); it++)
	{
		infile << *(it) << " ";
	}
	infile << endl;
	infile.close();
}

int main()
{
	IplImage* img=cvLoadImage("./test_images/face2.jpg", -1);
	IplImage* gray_img = cvCreateImage(cvGetSize(img), img->depth, 1);
	vector<pair<int, IplImage*>> pImg;
	vector<pair<int, IplImage*>>::iterator it;
	//getBGRpixels(img);
	//IplImage* canny_img = cvCreateImage(cvGetSize(img), img->depth, 1);
	//cvSmooth(gray_img, gray_img, CV_GAUSSIAN, 3, 0, 0);
	//IplImage* sobel_img = cvCreateImage(cvGetSize(gray_img), img->depth, 1);

	if (!img)
	{
		cout << "can not load image!" << endl;
	}
	else
	{
		cvCvtColor(img, gray_img, CV_BGR2GRAY);
		pImg=segementImage(gray_img);
		//cvCanny(gray_img, canny_img, 50, 180, 3);
		//IplImage* subImg = cvCreateImage()
		for (it = pImg.begin(); it < pImg.end(); it++)
		{
			IplImage* showImg = (*it).second;
			getGrayPixels(showImg);
			IplImage* canny_img = cvCreateImage(cvGetSize(showImg), img->depth, 1);
			cvCanny(showImg, canny_img, 50, 180, 3);
			int img_num = (*it).first;
			//cout << (*it).first << " ";
			char temp_path[30];
			string path = "./test_res/";
			sprintf(temp_path, "%s%d%s", "./test_res/face_two", img_num, ".txt");
			CvHistogram* hist = calcGrayHist(&showImg);
			//cout << canny_img->width << " " << canny_img->height <<endl;
			extractCornersAndCenter(canny_img, hist, temp_path);
			
			cvNamedWindow("img");
			cvShowImage("img", showImg);
			waitKey(0);
			cvDestroyWindow("img");
		}

		//运用canny和sobel算法提取边缘特征
		//cvCanny(img, canny_img, 50, 180, 3);
		//cvSobel(gray_img, sobel_img, 1, 1, 3);

		//cout << sobel_img->width << " " << sobel_img->height;
		//getGrayPixels(canny_img);
		//getGrayPixels(sobel_img);

		//cvNamedWindow("img");
		//cvShowImage("img", canny_img);
		//cvNamedWindow("img2");
		//cvShowImage("img2", sobel_img);
		//cvWaitKey(0);
		//cvDestroyWindow("img");
		//cvReleaseImage(&img);
		//cvReleaseImage(&gray_img);
		//cvReleaseImage(&canny_img);
		//cvReleaseImage(&sobel_img);

	}
	
	return 0;
}
