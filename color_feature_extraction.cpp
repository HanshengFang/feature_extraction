#include<iostream>
#include<cv.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<fstream>

using namespace std;
using namespace cv;


/*
@brief: 提取图片RGB三个通道的颜色信息并写入txt文件中
srcImg: 待提取特征的图片
features：用于保存图片特征的容器
*/
void colorFeatureExtraction(IplImage* srcImg,vector<float> features,string path)
{
	int R_bins = 8, G_bins = 8, B_bins = 8;//设置各通道bin的数量
	IplImage* r_plane = cvCreateImage(cvGetSize(srcImg), 8, 1);
	IplImage* g_plane = cvCreateImage(cvGetSize(srcImg), 8, 1);
	IplImage* b_plane = cvCreateImage(cvGetSize(srcImg), 8, 1);
	IplImage* planes[] = { r_plane, g_plane, b_plane };
	int hist_size[] = { R_bins, G_bins, B_bins };
	float r_ranges[] = { 0, 255 };//数组初始化
	float g_ranges[] = { 0, 255 };
	float b_ranges[] = { 0, 255 };
	float *ranges[] = { r_ranges, g_ranges, b_ranges };
	//cout << r_ranges << endl;
	//cout << *ranges << endl;
	cvCvtPixToPlane(srcImg, r_plane, g_plane, b_plane, 0);//分离多通道RGB值
	CvHistogram *hist = cvCreateHist(3, hist_size, CV_HIST_ARRAY, ranges, 1);
	cvClearHist(hist);
	cvCalcHist(planes, hist, 0, 0);
	features.clear();
	for (int r = 0; r < R_bins; r++)
	{
		for (int g = 0; g < G_bins; g++)
		{
			for (int b = 0; b < B_bins; b++)
			{
				float bin_val = cvQueryHistValue_3D(hist, r, g, b);
				features.push_back(bin_val);
			}
		}
	}

	//feature normalization
	float val_max = 1, val_min = 0;
	float feature_max = *max_element(features.begin(), features.end());
	float feature_min = *min_element(features.begin(), features.end());
	vector<float>::iterator it;
	for (it = features.begin(); it != features.end(); it++)
	{
		*it = ((*it) - feature_min) / (feature_max - feature_min + 1e-8);
		//cout << *it << " ";
	}

	ofstream infile;//写文件
	infile.open(path,ios::app);//以不覆盖的方式写文件
	for (it = features.begin();it != features.end(); it++)
	{
		infile << *it << " ";
		//cout << *it << " ";
	}
	infile << endl;
	infile.close();

}

int main()
{
	string file_path = ".//test_images//people";
	char temp[50];
	string ss;
	for (int i = 1; i < 4; i++)
	{
		sprintf(temp, "%d%s", i, ".jpg");
		ss = file_path+temp;
		IplImage *img = cvLoadImage(ss.c_str(), 1);
		vector<float> color_features;
		vector<float>::iterator fea_it;
		string path = "./color_feature.txt";
		if (!img)
		{
			cout << "fail to load image!" << endl;
		}
		else
		{
			colorFeatureExtraction(img, color_features, path);
		}
	}
	//IplImage *img = cvLoadImage("./people.jpg", 1);
	//vector<float> color_features;
	//vector<float>::iterator fea_it;
	//string path = "./color_feature.txt";

	//if (!img)
	//{
	//	cout << "fail to load image!" << endl;
	//}
	//else
	//{
	//	colorFeatureExtraction(img,color_features,path);
	//	
	//	/*cvNamedWindow("imgr");
	//	cvShowImage("imgr", r_plane);
	//	cvNamedWindow("imgg");
	//	cvShowImage("imgg", g_plane);
	//	cvNamedWindow("imgb");
	//	cvShowImage("imgb", r_plane);
	//	cvWaitKey(0);
	//	cvDestroyAllWindows();*/
	//}
	
	return 0;
}
