---
title: 基于OpenCV家用多摄像头监控设计
catalog: true
date: 2019-5-26 19:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- 图像算法
- c++
- 多线程
catagories: 
- c++
- 图像算法
- 并发编程
---

> 本篇博客主要针对家用多个摄像头监控入侵检测的问题，若有外人移动，则应把相应时刻图像抓取本地保存，但对于多个摄像头，不大可能用N个循环进行交替进行检测，这里主要用到多线程技术，每个相机创建一个线程单独进行监控，互不干扰，算法较简单，但较实用。从输出的nCounter可以看出那个相机采像较快。

```C
#include<iostream>
#include<thread>
#include<mutex>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define NumsCamera 2

cv::VideoCapture cap[NumsCamera];
cv::Mat src[NumsCamera];
deque<Mat> srcImgs[NumsCamera]; // 缓存队列，固定30
Ptr<BackgroundSubtractor> model[NumsCamera];
int nCounter[NumsCamera] = { 0 };//记录每个相机采集的总帧数

cv::Mat  foregroundMask;
int numImg = 0;
bool isESC = false;

void saveImage(int id,Mat& frame, Ptr<BackgroundSubtractor> model)
{
	model->apply(frame, foregroundMask, -1);
	//imshow(to_string(id), foregroundMask);
	
	vector<vector<cv::Point> > contours;
	// 查找轮廓，对应连通域
	cv::findContours(foregroundMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// 寻找最大连通域
	double maxArea = 500;
	vector<cv::Point> maxContour;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxContour = contours[i];
		}
	}

	// 将轮廓转为矩形框
	cv::Rect maxRect = cv::boundingRect(maxContour);
	imwrite(to_string(numImg++) + ".jpg", frame);
	cv::rectangle(frame, maxRect, Scalar(0, 255, 0), 2);
	imshow("detect"+ to_string(id), frame);
	waitKey(1);
}

void produce(int nCamera)
{
	double t1 = getTickCount();
	while (!isESC)
	{
		cap[nCamera] >> src[nCamera];
		saveImage(nCamera, src[nCamera].clone(), model[nCamera]);
		if (srcImgs[nCamera].size()<30)
		{
			nCounter[nCamera]++;
			srcImgs[nCamera].push_back(src[nCamera].clone());
		}
		else
		{
			srcImgs[nCamera].pop_front();
		}
		double t2 = getTickCount();
		string label = format("capture frames:%d, FPS:%.2f", nCounter[nCamera],getTickFrequency() / (t2 - t1));
		t1 = t2;
		putText(src[nCamera], label, Point(50, 100), 1, 2, Scalar(0, 0, 255));
		imshow("src" + to_string(nCamera), src[nCamera]);
		int key = waitKey(1);
		if (key == 27)
		{
			isESC = true; // 2个独立线程都结束
		}
		else if(key=='a')
		{
			imshow("src" + to_string(0), src[0]); // 第一个线程终止，第二个继续
			break;
		}
		else if(key == 'b')
		{
			imshow("src" + to_string(1), src[1]); // 第二个线程终止，第一个继续
			break;
		}
	}
}

int main()
{
	std::thread  producers[NumsCamera];
	for (size_t i = 0; i < NumsCamera; i++)
	{
		cap[i].open(i);
		model[i] = createBackgroundSubtractorMOG2();
		producers[i] = std::thread(produce,i);
	}

	producers[0].join();
	producers[1].join();
	return 0;
}
```

reference:https://www.lylinux.net/article/2018/1/24/38.html



