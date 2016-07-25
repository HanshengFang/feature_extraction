#include<cv.h>
#include<highgui.h>
#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;

int getHopCount(uchar i) //计算跳变次数
{
	int a[8] = { 0 };
	int k = 7;
	int cnt = 0;
	while (i)
	{
		a[k] = i & 1;
		i >>= 1;//右移一位
		--k;
	}
	for (int k = 0; k < 8; k++)
	{
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++cnt;
		}
	}
	return cnt;
}

void lbp59table(uchar* table)//降维数组 由256->59
{
	memset(table, 0, 256);
	uchar temp = 1;
	for (int i = 0; i < 256; i++)
	{
		if (getHopCount(i) <= 2)//跳变次数<=2的为非0值
		{
			table[i] = temp;
			temp++;
		}
		//cout << table[i] << endl;

	}
}

void LBP(IplImage* src, IplImage* dst)
{
	int width = src->width;
	int height = src->height;
	uchar table[256];
	lbp59table(table);
	for (int j = 1; j < width - 1; j++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			uchar neighbor[8] = { 0 };
			neighbor[7] = CV_IMAGE_ELEM(src, uchar, i - 1, j - 1);
			neighbor[6] = CV_IMAGE_ELEM(src, uchar, i - 1, j);
			neighbor[5] = CV_IMAGE_ELEM(src, uchar, i - 1, j + 1);
			neighbor[4] = CV_IMAGE_ELEM(src, uchar, i, j + 1);
			neighbor[3] = CV_IMAGE_ELEM(src, uchar, i + 1, j + 1);
			neighbor[2] = CV_IMAGE_ELEM(src, uchar, i + 1, j);
			neighbor[1] = CV_IMAGE_ELEM(src, uchar, i + 1, j - 1);
			neighbor[0] = CV_IMAGE_ELEM(src, uchar, i, j - 1);
			uchar center = CV_IMAGE_ELEM(src, uchar, i, j);
			uchar temp = 0;
			for (int k = 0; k < 8; k++)
			{
				temp += (neighbor[k] >= center) << k;//计算LBP的值
			}
			//CV_IMAGE_ELEM(dst, uchar, i, j) = temp;
			CV_IMAGE_ELEM(dst, uchar, i, j) = table[temp];//降到59维
		}
	}
	
}

void FillWhite(IplImage *pImage)
{
	cvRectangle(pImage, cvPoint(0, 0), cvPoint(pImage->width, pImage->height), CV_RGB(255, 255, 255), CV_FILLED);
}

//创建灰度图像的直方图
CvHistogram* CreateGrayImageHist(IplImage **ppImage)
{
	int nHistSize = 256;
	float fRange[] = { 0, 255 };//灰度级范围
	float *pfRanges[] = { fRange };
	CvHistogram *pcvHistogram = cvCreateHist(1, &nHistSize, CV_HIST_ARRAY, pfRanges);
	cvCalcHist(ppImage, pcvHistogram);
	return pcvHistogram;
}

//根据直方图创建直方图图像
IplImage* CreateHistogramImage(int nImageWidth, int nScale, int nImageHeight, CvHistogram *pcvHistogram)
{
	IplImage *pHistImage = cvCreateImage(cvSize(nImageWidth*nScale, nImageHeight), IPL_DEPTH_8U, 1);
	FillWhite(pHistImage);

	//统计直方图中最大的直方块
	float fMaxHistValue = 0;
	cvGetMinMaxHistValue(pcvHistogram, NULL, &fMaxHistValue, NULL, NULL);

	//分别将每个直方图的值绘制到图中
	int i;
	for (int i = 0; i < nImageWidth; i++)
	{
		float fHistValue = cvQueryHistValue_1D(pcvHistogram, i);//像素为i直方块大小
		int nRealHeight = cvRound((fHistValue / fMaxHistValue)*nImageHeight);//要绘制的高度
		cvRectangle(pHistImage, cvPoint(i*nScale, nImageHeight - 1), cvPoint((i + 1)*nScale - 1, nImageHeight - nRealHeight),
			cvScalar(i, 0, 0, 0), CV_FILLED);

	}
	return pHistImage;
}

void extractPixelLBP(IplImage* srcImg,string path)
{
	ofstream myfile;
	myfile.open(path);
	if (!myfile)
	{
		cout << "can not open the file,exit" << endl;
		exit(1);
	}
	else
	{
		int temp;
		//cout << lbp_img->height << " " << lbp_img->width << endl;
		for (int i = 0; i < srcImg->height; i++)
		{
			for (int j = 0; j < srcImg->width; j++)
			{
				temp = cvGet2D(srcImg, i, j).val[0];
				//cout << temp << endl;
				myfile << temp << " ";
			}
			myfile << endl;
		}
	}
	myfile.close();
}

float getMaxMin(float val, float maxVal, float minVal)
{
	if (val > maxVal)
	{
		maxVal = val;
	}
	else if (val<minVal)
	{
		minVal =val;
	}
	return maxVal, minVal;
}

float normalizePixelValue(float val, float maxVal, float minVal)
{
	float gap = maxVal - minVal;
	if (gap < 1)
	{
		return 0;
	}
	else
	{
		val=val / gap;
		return val;
	}
}

void savePixel(string path, float val)
{
	ofstream infile;
	infile.open(path,ios::app);//以不覆盖的方式写文件
	if (!infile)
	{
		cout << "fail to load file!"<<endl;
		exit(0);
	}
	infile <<val << " ";
	infile.close();
}

void changeLine(string path)//给每张图片特征换行，便于分析
{
	ofstream infile;
	infile.open(path, ios::app);//以不覆盖的方式写文件
	if (!infile)
	{
		cout << "fail to load file!" << endl;
		exit(0);
	}
	infile <<endl;
	infile.close();
}

int main()
{
	string filename = "G:\\face_datasets\\att_faces\\s";
	string path_feature1 = ".\\feature_train.txt";
	string path_feature2 = ".\\feature_test.txt";
	string path_feature;
	char temp[100];
	string ss;
	string filePath;
	for (int i = 1; i < 41; i++)
	{
		for (int j = 1; j < 11; j++)
		{
			sprintf(temp, "%d%s%d%s", i,"\\", j,".pgm");
			ss = temp;
			filePath = filename + ss;
			IplImage* img = cvLoadImage(filePath.c_str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
			if (!img)
			{
				cout << "fail to load images" << endl;
			}
			else
			{
				IplImage* lbp_img = cvCreateImage(cvSize(img->width, img->height), 8, 1);//分配空间
				//IplImage* gray_img = cvCreateImage(cvSize(img->width, img->height), 8, 1);
				//cvCvtColor(img, gray_img, CV_BGR2GRAY);
				LBP(img, lbp_img);//LBP算法提取特征

				CvHistogram* pcvHistogram = CreateGrayImageHist(&lbp_img);//建立图像的灰度直方图
				int h_bins = 16, s_bins = 4;
				float bin_val=0;
				vector<float>vec;
				vector<float>::iterator it;
				for (int h = 0; h < h_bins; h++)
				{
					for (int s = 0; s < s_bins; s++)
					{
						int i = h*s_bins + s;
						bin_val = cvQueryHistValue_1D(pcvHistogram, i);
						//cout << bin_val << " ";
						vec.push_back(bin_val);
					}
				}
				//不归一化的特征
				/*for (it = vec.begin(); it != vec.end(); it++)
				{
					savePixel(path_feature2, *it);
				}*/

				if (j == 1 || j == 3 || j == 5)
				{
					path_feature = path_feature2;
				}
				else
				{
					path_feature = path_feature1;
				}
				float max_bin = *(vec.begin());
				float min_bin = *(vec.begin());

				for (it = vec.begin(); it != vec.end(); it++)
				{
					max_bin, min_bin = getMaxMin((*it), max_bin, min_bin);
				}

				for (it = vec.begin(); it != vec.end(); it++)
				{
					bin_val = normalizePixelValue((*it), max_bin, min_bin);//归一化
					savePixel(path_feature, bin_val);
					//cout << bin_val << " ";
				}
				savePixel(path_feature, i);//添加训练集的label信息
				changeLine(path_feature);//每一张人脸用一行来存放特征

				/*cvNamedWindow("img");
				cvShowImage("img", img);
				cvWaitKey(0);
				cvDestroyWindow("img");*/

				vec.clear();
				cvReleaseImage(&img);
				cvReleaseImage(&lbp_img);
				cvReleaseHist(&pcvHistogram);
			}	
		}
	}

	/*
	//extractPixelLBP(lbp_img, path);//提取LBP图像像素值
	//创建直方图图像
	int nHistImageWidth = 255;
	int nHistImageHeight = 150;
	int nScale = 2;
	IplImage *pHistogram = CreateHistogramImage(nHistImageWidth, nScale, nHistImageHeight, pcvHistogram);
	cvNamedWindow("src image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("gray Image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("hist image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("lbp image", CV_WINDOW_AUTOSIZE);

	cvShowImage("src image", img);
	cvShowImage("gray Image", gray_img);
	cvShowImage("hist image", pHistogram);
	cvShowImage("lbp image", lbp_img);

	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&img);
	cvReleaseImage(&lbp_img);
	cvReleaseImage(&gray_img);
	cvReleaseImage(&pHistogram);
	//cvReleaseHist(&pcvHistogram);
	*/

	return 0;
}
