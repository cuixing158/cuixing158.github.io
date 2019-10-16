---
title: C++多线程函数之异步执行微观
catalog: true
date: 2019-10-16 18:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- c++
- 多线程

catagories: 
- c++
- 并发编程
---

> 利用多线程编程有时候可以极大提高程序的执行效率，相比串行编程，涉及的逻辑机制不易理解，但能够带来效率的提升！在某些场合更适合并行计算，如
多摄像机的采像处理要求同步，另外各个相机在某种算法下独立（子线程）识别到了某个动作发生，就应当返回给主线程。各个子线程有时也需要同步或者一些
条件加以限制，比如各个动作识别之间的间隔不应该低于50帧，要在线程之间共享某些变量来限制，下面的程序模拟了该思想过程，从结果上看，可以很好的完成
了任务。

```c++
// author:cuixingxing
// email: cuixingxing150@gmail.com
// 2019.9.29
// 测试thread线程函数改变值/重复的情况

#include<iostream>
#include<thread>
#include<fstream>
#include< sstream> //stringstream

using namespace std;

class A
{
public:
	A()
	{
		m_fid.open("out.txt");
		m_last_num = 0;
		m_num = 0;
	};
	~A()
	{
		m_fid.close();
	}

	void getSequence(int in[],int index,bool &out)
	{
		// 线程函数里面要用随机种子产生随机数，种子可以采用各自的线程id,否则rand()都是一样的结果
		std::stringstream ss;
		ss << std::this_thread::get_id();
		unsigned int id = std::stoull(ss.str()); // 线程函数转换为unsigned int
		srand(id);// 当前id做为随机种子

		out = false;// 初始false
		int randnum =  rand() % 3;
		printf("%d\n",randnum);
		out = randnum;
		if ((out==0)&&(m_num-m_last_num>=50))
		{
			out = true;
			m_last_num = m_num;
		}
		else
		{
			out = false;
		}
	}

	void recog3D(int in[])
	{
		m_num++;
		bool isGet[4] = {false};
		thread t1(&A::getSequence, this, in, 0, std::ref(isGet[0])); // 线程不能为同一个this对象，否则里面成员变量资源争夺,加互斥锁可以解决；此示例中成员变量资源共享，4个线程函数只要任何一个有发生（isGet某一个为true时）每次发生间隔保证至少50帧以上！
		thread t2(&A::getSequence, this, in, 1, std::ref(isGet[1]));
		thread t3(&A::getSequence, this, in, 2, std::ref(isGet[2]));
		thread t4(&A::getSequence, this, in, 3, std::ref(isGet[3]));
		t1.join();
		t2.join();
		t3.join();
		t4.join();
		m_fid << "numframe:" << m_num << ",isGet:a" << isGet[0] << ",b" << isGet[1] << ",c" << isGet[2] << ",d" << isGet[3] << endl;
	}

private:
	ofstream m_fid;
	int m_last_num;
	int m_num;
};
		

void main()
{
	A a;
	int in[4] = { 1,2,3,4 };

	for (size_t i = 0; i < 2000; i++)
	{
		a.recog3D(in);
	}

}
```
**注意:** 多线程函数要产生随机变量并非那么直接，因为随机种子数问题，很可能导致各个线程产生的都是同样的一组数据，上面使用当前线程的id作为随机数，
所以randnum输出的数据基本是无序的。另外从输出out.txt中看出各个动作至少相隔50帧才发生一次，动作的随机过程模拟了实际动作发生的时间。 