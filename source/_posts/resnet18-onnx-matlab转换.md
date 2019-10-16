---
title: resnet18-onnx-matlab转换
catalog: true
date: 2019-09-25 08:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- pytorch
- onnx
- matlab
catagories: 
- pytorch
- 深度学习
---

> 本博客主要验证pytorch torchvision中官方预训练的模型导入matlab中看是否成功，以resnet18为例，中间格式通过onnx进行转换。主要分为python实现、onnx导出、matlab导入、
matlab识别。结论：转换识别成功

# 一、python中实现简单demo
代码主要以一个摄像头采集的图像识别为例，采集识别OK后，转成onnx通用模型，Python实现代码及导出代码如下：
```python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:55:17 2019
% author:cuixingxing
% email: cuixingxing150@gmail.com
% 2019.9.29
%

@author: Administrator
"""

import torch
import torchvision
from torchvision import transforms
import cv2

size = (224,224)
labelpath = r'./synset_words.txt'

def readLabels(labelpath=labelpath):
    with open(labelpath) as fid:
        labels = fid.readlines()
        return labels
    
def classifyImg(net,img):
    net.to('cuda').eval()
    img = cv2.resize(img,size)
    trans = transforms.Compose([transforms.ToTensor()]) # ToTensor()可以把PIL图像或者opencv图像（BGR [0,255],H*W*C）直接转换为pytorch可接收的tensor类型（RGB,[0,1],C*H*W）的格式
    tensorImg = trans(img).unsqueeze(0).to('cuda')
    
    out = net(tensorImg) # out 是一个二维矩阵
    scores,idxs = torch.sort(out,dim=1,descending=True)
    labels = readLabels(labelpath)
    for i in range(1):
        print("top {:d},predict score:{:.5f},label:{:s}".format(i,scores[0][i],labels[idxs[0][i]]))

if __name__=='__main__':
    model = torchvision.models.resnet18(True)
    cap = cv2.VideoCapture(0)
    isRead,img = cap.read()
    while isRead:
        isRead,img = cap.read()
        classifyImg(model,img)
        cv2.imshow("",img)
        key = cv2.waitKey(10)
        if key==27:
            break
        if key ==' ':
            cv2.waitKey()
            
    # %% 导出为onnx，并导入到matlab中运行查看
    torch.onnx.export(model,torch.rand(1,3,224,224).to('cuda'),'resnet18_Torch.onnx',verbose=True)
```
![resnet18-onnx-matlab转换](python_rec.png)

# 二、Matlab加载实现
对于上一步中生成的resnet18_Torch.onnx，加载到matlab中，实现代码如下：
```matlab
% author:cuixingxing
% email: cuixingxing150@gmail.com
% 2019.9.29
%
net = importONNXNetwork('E:\python_work\resnet18Export.onnx','OutputLayerType','classification');
path = "E:/python_work/synset_words.txt";

%% predict
cap = webcam(2);
classes = getClasses(path);

depVideoPlayer = vision.DeployableVideoPlayer;
oriImg = snapshot(cap);
depVideoPlayer(oriImg);% 显示img
while isOpen(depVideoPlayer)
    oriImg = snapshot(cap);
    img = imresize(oriImg,[224,224]);
    tensorImg = im2double(img);
    %% 
    Ypredict = predict(net,tensorImg);
    [max_val,ind] = max(Ypredict);
    predictLabel = classes(ind);
    predictScore = max_val;
    str = sprintf('predictLabel:%s, predictScore:%.5f\n',string(predictLabel),string(predictScore));
    RGB = insertText(oriImg,[10,20],str);
    depVideoPlayer(RGB); % 显示RGB
end

function classes = getClasses(labelsPath)
fid = fopen(labelsPath,'r');
s = textscan(fid,'%s', 'Delimiter',{'    '});
s = s{1};
% classes =cellfun(@(x)x(11:end),s,'UniformOutput',false);
classes = s;
fclose(fid);
end

```
![resnet18-onnx-matlab转换](matlab_rec.png)
从图中可以看出，同一个模型情况下，识别结果一致，注意*网络输入图像大小、顺序、类型要一致！*
