---
title: Why are the features obtained by "predict" and "forward" function very different?
catalog: true
date: 2020-07-11 08:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- batchnorm
- Basic concepts
catagories: 
- matlab
- algorithm
---
    
&#160; &#160; &#160; &#160;本博客主要内容本应该是我提出的issue被广泛讨论的，所以写成英文方便国外网友讨论，后来干脆整理成了一篇博客，依旧保留了英文模式，逐步引人入思，遂成文，发之，英文表达有不妥之处，敬请谅解！ 
> According to official instructions, "[forward](https://ww2.mathworks.cn/help/deeplearning/ref/dlnetwork.forward.html)" is used for inference during the network training phase, and "[predict](https://ww2.mathworks.cn/help/deeplearning/ref/dlnetwork.predict.html)" is used for inference during the network prediction phase.But there are a large number of networks used by the batchnorm layer,the features of forward and predict are very different.


&#160; &#160; &#160; &#160;For convenience, only observe the output features  of the first  convolution layer and bathhorm layer of resnet50,We first take the [resnet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html?s_tid=srchtitle) network provided by matlab as an example to illustrate, convert to [dlnetwork](https://www.mathworks.com/help/deeplearning/ref/dlnetwork.html?s_tid=srchtitle) and compare.

```matlab
net50 = resnet50;
inputsize = [224,224];
img = imresize(imread('peppers.png'),inputsize);
inputImg = dlarray(im2single(img),'SSCB');
lg = layerGraph(net50);
newlg = removeLayers(lg,'ClassificationLayer_fc1000');
dlnet = dlnetwork(newlg); % take long time ???
```

![fig](image_0.png)
<center>network</center>

## simple case:
We extract the feature named "fc1000_softmax" layer，big different features between forward_f and predict_f ?

```matlab
[forward_f,state1] = forward(dlnet,inputImg);
[~,ind1] = max(forward_f)
```


```text:Output
ind1 = 
  1(C) x 1(B) dlarray
   464
```


