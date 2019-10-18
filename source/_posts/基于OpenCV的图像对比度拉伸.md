---
title: 基于OpenCV的图像对比度拉伸
catalog: true
date: 2019-10-18 17:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- 图像算法
- c++
catagories: 
- c++
- 图像算法
---

> 基于图像的线性对比度拉伸/增强绝大多数教程都在论述基于范围的拉伸，本博客主要提出基于均值的对比度拉伸，相比于前者，更能注重
给定均值附近的像素的增强，而不是全局[0,255]范围内增强。最后给出两种算法的代码实现过程，并比较手写代码与自带代码的性能。

假如有一副比较暗的图像，亮度像素等级普遍在10左右，而我希望的图像均值avg2=25左右波动，有一种很简单的
想法是直接把原图像所有像素加上15就可以了，但是问题是对比度变得比较差！这时候就需要图像拉伸/增强。下面给出算法步骤：

**1**、设原始图像矩阵为A=A(x,y)，其像素范围在[minV,maxV],均值为avg1，目标平均均值为C。
则变换到均值为C的增强矩阵B(x,y)=A(x,y)+C-avg1,其像素范围为[minV+C-avg1,maxV+C-avg1]
**2**、计算拉伸因子alpha，根据尽可能把像素拉伸到最大范围（对比度达到最大），分两种情况计算并比较，
在均值C不变的情况下，最小值minV+C-avg1到C之间的拉伸因子为alhpa1=(C-0)/(C-(minV+C-avg1));同理，
C到最大值maxV+c-avg1之间的拉伸因子为alhpa2=(255-C)/(maxV+C-avg1-C).比较alhpa1与alpha2的大小，取较小者
为alpha。
**3**、设最终图像对比度拉伸/增强的矩阵为D = D(x,y)，Bmin = minV+C-avg1, Bmax = maxV+C-avg1,则根据线性变换公式，有(C-B(x,y))/(C-Bmin)==(C-D(x,y))/(alpha*(C-Bmin)),
最终可以解得D(x,y) = C-alpha*(C-B(x,y))

另一种基于给定范围的对比度教程较多，不具体阐述，两种对比度拉伸都给出OpenCV实现代码，如下所示：

```C
// 图像对比度拉伸
#include<iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

inline float calcAvg(const Mat& src, const Rect&rect)
{
	Mat image = src.clone();
	if (image.channels() == 3)
	{
		cvtColor(image, image, COLOR_BGR2GRAY);//,CV_BGR2GRAY
	}

	if ((rect & Rect(0, 0, image.cols, image.rows)) == rect)
	{
		Mat patch = image(rect);
		patch.convertTo(patch, CV_32F);
		float avg_value = 0.0;
		for (size_t i = 0; i < patch.rows; i++)
		{
			float *data = patch.ptr<float>(i);
			for (size_t j = 0; j < patch.cols; j++)
			{
				avg_value += (*data++);
			}
		}
		return avg_value*1.0 / (patch.rows*patch.cols);
	}
	else
	{
		return 0.0;
	}
}

void main()
{
	VideoCapture cap(0);
	Mat src;
	
	while (cap.read(src))
	{
		Mat ori = src.clone();
		Mat image = src.clone();
		Mat dst;
		// 一、基于给定范围的线性对比度拉伸
		// 1、手写函数，即使是指针，耗时相对自带函数较长
		Vec3b *pDataMat;
		int pixMax = 0, pixMin = 255;
		int targetpixMax = 200, targetpixMin = 100;

		//计算图像像素的最大值和最小值
		double t1 = getTickCount();
		for (int i = 0; i < image.rows; i++)
		{
			pDataMat = image.ptr<Vec3b>(i);
			for (int j = 0; j < image.cols; j++)
			{
				for (int k = 0; k < image.channels(); k++)
				{
					if (pDataMat[j][k] > pixMax)
						pixMax = pDataMat[j][k];
					if ((int)pDataMat[j][k] < pixMin)
						pixMin = (int)pDataMat[j][k];
				}
			}
		}
		//cout << pixMax << "," << pixMin << endl;
		
		for (int i = 0; i < image.rows; i++)
		{
			pDataMat = image.ptr<Vec3b>(i);
			for (int j = 0; j < image.cols; j++)
			{
				for (int k = 0; k < image.channels(); k++)
				{
					pDataMat[j][k] = (pDataMat[j][k] - pixMin) * (targetpixMax-targetpixMin) / (pixMax - pixMin)+targetpixMin;
				}
			}
		}
		double t2 = getTickCount();
		putText(image, "time:" + to_string( (t2 - t1)/ getTickFrequency()*1000), Point(20, 30), 1, 2, Scalar(0, 255, 0), 2); // 3.6 ms
		imshow("基于给定范围的图像拉伸（指针手写）", image);

		// 2、系统自带函数
		Mat dstImage;
		double t3 = getTickCount();
		ori.convertTo(dstImage, -1, 100.0/255, 100);
		double t4 = getTickCount();
		putText(dstImage, "time:" + to_string((t4 - t3) / getTickFrequency() * 1000), Point(20, 30), 1, 2, Scalar(0, 255, 0), 2); // 0.3ms
		imshow("基于给定范围的图像拉伸（opencv自带）", dstImage);

		// 二、基于给定平均亮度的线性对比度拉伸，尽可能自动映射到[0,255]范围内
		// 计算图像均值
		double t5 = getTickCount();
		float grayScale = calcAvg(src, Rect(0, 0, src.cols, src.rows));
		double t6 = getTickCount();
		printf("calcAvg time:%.2f\n " ,(t6 - t5) / getTickFrequency() * 1000); // 用指针手写也要1.54ms

		double t7 = getTickCount();
		Scalar mean_val = cv::mean(src);
		float fmean = (mean_val[0] + mean_val[1] + mean_val[2])/3.0;
		double t8 = getTickCount();
		printf( "mean time:%.2f\n" , (t8 - t7) / getTickFrequency() * 1000); // 自带仅需要0.2ms

		float avePix = 100; //给定一个图像均值
		float delta = 3;//误差容忍因子
		Mat temp = src.reshape(1);
		double minVal, maxVal;
		minMaxIdx(temp, &minVal, &maxVal, NULL, NULL);
		if ((fmean<avePix- delta) || (fmean>avePix+ delta))  // 每个原图像调整到均值avePix左右
		{
			src.convertTo(src, CV_32F);
			src = src + Scalar::all(avePix) - mean_val;
			float alpha1 = (avePix - 0) / (fmean - minVal);
			float alpha2 = (255 - avePix) / (maxVal - fmean);
			float alpha = (alpha1 < alpha2) ? alpha1 : alpha2;
			dst = Scalar::all(avePix) - alpha*(Scalar::all(avePix) - src); // 如果是CV_8U类型，数据会溢出！！！
			dst.convertTo(dst, CV_8U);
			imshow("基于均值的线性拉伸", dst);
		}
		cout << "ave:" << mean(dst)  << endl<<endl;
		imshow("原始图像", ori);
		waitKey(10);
	}
}
```

reference:https://blog.csdn.net/xiahouzuoxin/article/details/26478179



