#include<iostream>
#include<cv.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>
#include<fstream>

using namespace std;
using namespace cv;

void printMat(CvMat* mat)//打印mat矩阵
{
	for (int i = 0; i < mat->rows; i++)
	{
		for (int j = 0; j < mat->cols; j++)
		{
			cout << cvmGet(mat, i, j)<<" ";
		}
		cout << endl;
	}
}

/*
CvMat* extractTrainAndTest(CvMat* Data_labels, CvMat* Data, CvMat* Labels, string path)
	   CvMat* Data_labels：提取出的包含label信息的数据集
       CvMat* Data：从Data_labels分离出的不包含label信息的数据集
       CvMat* Labels：从Data_labels分离出的label数据集
       string path：txt数据集文件的读取路径
     return: Data,Labels;
*/
CvMat* extractTrainAndTest(CvMat* Data_labels, CvMat* Data, CvMat* Labels, string path)
{
	ifstream readfile;//读取原始文件
	readfile.open(path);
	for (int i = 0; i < Data_labels->rows; i++)
	{
		for (int j = 0; j <Data_labels->cols; j++)
		{
			readfile >> Data_labels->data.fl[i*(Data_labels->cols) + j];//数据写入trainData矩阵中(包含label信息)
			//cout << trainData->data.fl[i*(trainData->cols) + j];
		}
	}
	readfile.close();

	//从Data_labels中提取label并存放于Labels中
	for (int i = 0; i < Labels->rows; i++)
	{
		for (int j = 0; j < Labels->cols; j++)
		{
			Labels->data.fl[i*(Labels->cols) + j] = Data_labels->data.fl[i*(Data_labels->cols) + 64];
		}
	}

	//分离Data_labels中的Data信息
	for (int i = 0; i < Data->rows; i++)
	{
		for (int j = 0; j < Data_labels->cols; j++)
		{
			Data->data.fl[i*(Data->cols) + j] = Data_labels->data.fl[i*(Data_labels->cols) + j];
		}
	}

	return Data, Labels;
}

int main()
{

	string file_path_train = "./feature_train.txt";//训练集txt文件路径
	string file_path_test = "./feature_test.txt";//测试集txt文件路径

	const int K = 3;//knn算法中K值得设定
	int train_sample_rows = 30;
	int train_sample_cols = 65;

	int test_sample_rows = 9;
	int test_sample_cols = 65;

	//for training
	CvMat* trainData_labels = cvCreateMat(train_sample_rows, train_sample_cols, CV_32FC1);//创建训练集(带label)测试集并初始化
	CvMat* trainClass = cvCreateMat(train_sample_rows, 1, CV_32FC1);//原始的label存放在trainData中，需要分割
	CvMat* trainData = cvCreateMat(train_sample_rows, train_sample_cols - 1, CV_32FC1);
	cvZero(trainData_labels);
	cvZero(trainClass);
	cvZero(trainData);
	trainData, trainClass = extractTrainAndTest(trainData_labels, trainData, trainClass, file_path_train);
	//printMat(trainClass);
	//printMat(trainData);

	//for testing
	CvMat* testData_labels = cvCreateMat(test_sample_rows, test_sample_cols, CV_32FC1);//创建训练集(带label)测试集并初始化
	CvMat* testClass = cvCreateMat(test_sample_rows, 1, CV_32FC1);//原始的label存放在trainData中，需要分割
	CvMat* testData = cvCreateMat(test_sample_rows, test_sample_cols - 1, CV_32FC1);
	cvZero(testData_labels);
	cvZero(testClass);
	cvZero(testData);
	testData, testClass = extractTrainAndTest(testData_labels, testData, testClass, file_path_test);
	//printMat(testData);
	//printMat(testClass);

	/*ifstream readfile;//读训练集测试集文件
	readfile.open(file_path_train);
	//float temp[2][64];
	for (int i = 0; i < trainData_labels->rows; i++)
	{
		for (int j = 0; j < trainData_labels->cols; j++)
		{
			readfile >>trainData_labels->data.fl[i*(trainData_labels->cols)+j];//数据写入trainData矩阵中(包含label信息)
			//cout << trainData->data.fl[i*(trainData->cols) + j];
		}
	}
	//cout << "finished!" << endl;
	readfile.close();
	//printMat(trainData);

	//从trainData中提取label并存放于trainClass中
	for (int i = 0; i < trainClass->rows; i++)
	{
		for (int j = 0; j < trainClass->cols;j++)
		{
			trainClass->data.fl[i*(trainClass->cols) + j] = trainData_labels->data.fl[i*(trainData_labels->cols) + 64];
		}
	}
	//printMat(trainClass);

	//分离trainData中的label信息
	for (int i = 0; i < trainData->rows; i++)
	{
		for (int j = 0; j < trainData_labels->cols; j++)
		{
			trainData->data.fl[i*(trainData->cols) + j] = trainData_labels->data.fl[i*(trainData_labels->cols) + j];
		}
	}
	//printMat(trainData);*/

	//训练模型knn
	CvKNearest knn(trainData, trainClass, 0, false, 32);
	//cout << trainData->cols << " " << trainClass->cols << endl;
	CvMat* nearest = cvCreateMat(testClass->rows,K, CV_32FC1);//表示K个最近邻样本的响应值
	float res[9];
	int  accuracy = 0;
	/*for (int i = 0; i < test_sample_rows; i++)
	{
		res = knn.find_nearest(testData, K, 0, 0, nearest, 0);
		for (int k = 0; k < K; k++)
		{
			cout << "nearest: " << nearest->data.fl[k] << " ";
			cout << "test label: " << testClass->data.fl[i*(testClass->cols) + k] << endl;
			if (nearest->data.fl[k] == testClass->data.fl[i*(testClass->cols)+k])
				accuracy++;
		}
	}*/
	/*for (int i = 0; i < 9; i++)
	{
		res[i] = knn.find_nearest(testData, K, 0, 0, nearest, 0);
		cout << res[i] << " ";
	}
	cout << endl;*/
	float r = knn.find_nearest(testData, K, 0, 0, nearest, 0);
	cout << r << endl;
	for (int i = 0; i < K; i++)
	{
		cout << " " << nearest->data.fl[i];
	}
	/*cout << "finished..." << endl;
	cout << K << "nearest response:" << endl;
	for (int i = 0; i < K; i++)
	{
		cout << " " << nearest->data.fl[i];
	}*/
	//释放内存
	cvReleaseMat(&trainData_labels);
	cvReleaseMat(&trainClass);
	cvReleaseMat(&trainData);
	cvReleaseMat(&testData_labels);
	cvReleaseMat(&testClass);
	cvReleaseMat(&testData);
	return 0;
}
