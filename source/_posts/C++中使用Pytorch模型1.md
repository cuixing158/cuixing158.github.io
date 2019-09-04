---
title: C++中使用Pytorch模型1
catalog: true
date: 2018-11-25 08:25:53
subtitle: 
header-img: "sea.jpg"
tags: 
- pytorch
- c++
catagories: 
- pytorch
- 深度学习
---

>最近pytorch官方更新了稳定版本的1.0版本，相较于0.4版本，主要改进在于部署方面，这次参考官方给的示例程序（https://pytorch.org/tutorials/advanced/cpp_export.html），在Windows 平台，VS2015环境下搭建了环境，成功在C++中使用，现做如下记录：

&#160; &#160; &#160; &#160;根据官方描述，在pytorch中训练好的模型要在C++中使用，主要有tracing和annotation两种方式，这2种方式都是把模型转换成TorchScript，具体不阐述，详见官方网站
本次使用tracing方式，配置环境：win10+pytorch1.0+anaconda+spyder+vs2015

1、	首先搭建好python环境，最好使用新版本的python，现在官方各种版本都支持了，太好了。一句命令可以安装好pytorch1.0.
2、	安装好上面所有环境后，写个脚本测试下模型输入和输出。
![Fig1](图片1.png)
运行无误后（有错误肯定是没安装好pytorch，多试试），会在”pytorch1_0.py”所在目录生成model.pt模型参数文件。
输出窗口：
![Fig2](图片2.png)
3、	官方主页上下载好libtorch，该库独立于python存在
![Fig3](图片3.png)
下载解压后有5个文件夹，中间example-app是自己建的示例程序（其他位置也可以），如上图所示。
4、	在example-app中建立example-app.cpp和CMakeLists.txt，按照官方写入内容做就是了，然后在CMake-gui下配置好环境，我的如下：
![Fig4](图片4.png)
Configure后会有错误提示，因为CMake找不到torch的位置，所以添加Torch_DIR的value为
![Fig5](图片5.png)
重新Configure,这次没有上面那个错误了，但是会出现https://github.com/pytorch/pytorch/issues/14951这个上面说的错误。那就改CMAKE_BUILD_TYPE的值，重新制定Debug或者Release，
![Fig6](图片6.png)
添加了这一句后，保存，重新configure，OK，成功
![Fig7](图片7.png)
忽略警告即可，生成Generate，在build文件夹下生成如下项目文件。
![Fig8](图片8.png)
5、	VS2015打开example-app.sln工程，设定example-app项目为启动项目（右键单击设定启动项），如果没有错误，会有如下结果。
![Fig9](图片9.png)
跟上面python环境运行一致。


