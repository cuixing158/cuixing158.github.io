---
title: pytorch图像分类
catalog: true
date: 2018-10-05 12:25:53
subtitle: 
header-img: "photo.jpg"
tags: 
- 图像分类
- onnx
catagories: 
- pytorch
- 深度学习
---

> 利用深度学习对图像分类是非常普遍的问题，原来用过[SVM+HOG](https://blog.csdn.net/cuixing001/article/details/70908064)传统方法,[深度学习Alexnet网络对图像进行分类/预测（迁移学习）](https://blog.csdn.net/cuixing001/article/details/75807845)进行识别，效果还不错，这次改用pytorch框架加以实现。
>


# 数据准备
---
数据集依然来自于上面链接的博客采用的[图像](https://pan.baidu.com/s/1i5OhC7z)(百度网盘提取码:utn7)，该图像集有5个类别，每个类别大概40张左右。在电脑中存储的格式如下：
![trainimgs](trainimgs.jpg)

测试数据集保存在另外的一个文件夹，如下图所示：
![testimgs](testimgs.jpg)

## 数据输入
---
为了把图像读入到网络，先把图像路径和类别写到txt中，注意图像路径位置，脚本代码`getImgNames.py`为：
```python
import os
rootall = r'F:\imagesData\svm_images'
train_path = r'F:\imagesData\svm_images\train_images'
test_path = r'F:\imagesData\svm_images\test_image'
#ball_names = [name for name in all_names if name.endswith('_ball.jpg')]
#bgm_names = [name for name in all_names if name.endswith('_bgm.jpg')] 
           
#%% train.txt
fid = open(os.path.join(rootall,'train.txt'),'w')
n = 1
for root, dirs, files in os.walk(train_path):
    if n==1:
        folder_names = dirs
    for name in files:
        folder,subfolder = os.path.split(root)
        index = folder_names.index(subfolder)
        fid.write(os.path.join(root, name)+'  '+str(index)+'\n')
    n = n+1
fid.close()
```
train.txt中写的图像路径和类别为：
![44](44.jpg)

pytorch中提供了Dataset,DataLoader类专门处理数据的输入问题，用户数据一般继承Dataset,实现__init__()，__getitem__()，__len__()方法即可使用。
代码文件为`train_save.py`
```python
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from PIL import Image
#import onnx

root1=r"F:\imagesData\svm_images"
input_size = (3,300,300)

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()            
            words = line.split()    
            imgs.append((words[0],int(words[1])))
            
        self.imgs = imgs
        self.transform = transforms.Compose([transforms.Resize(input_size[1:]),
                                             transforms.ToTensor()])
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=os.path.join(root1,'train.txt'), transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=40, shuffle=True)
```

## 网络结构和训练
---
由于图像比较少，设计了简单的网络就可以识别分类了，注意全连层`torch.nn.Linear`的输入计算，否则会出现维度不匹配的情况。
```python
#%% 网络结构和训练
class testNet(nn.Module):
    def __init__(self,input_size=input_size):
        super(testNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        
        n_size = self._get_linear_inNums(input_size)#根据输入图像大小自动推断fc层输入
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(n_size , 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),        
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(), 
            torch.nn.Linear(10, 5)
            )
        
    def _get_linear_inNums(self,shape):
        batch_x = 1
        temp = Variable(torch.rand(batch_x,*shape))
        single_feature = self._forward_features(temp)
        n_size = single_feature.view(batch_x,-1).size(1)
        return n_size
    
    def _forward_features(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        out = self._forward_features(x)
        print('out.size():',out.size())
        res = out.view(out.size(0),-1)
        out = self.dense(res)
        out =  F.log_softmax(out,dim = 1)
        return out
    
net = testNet()
#net.cuda() #把网络推送到GPU

# %% train
#print(net)
optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.CrossEntropyLoss()
#torch.cuda.set_device(0)
#loss_func.cuda() # 损失函数推送到GPU

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
#        batch_x  = batch_x.cuda() # 推送到GPU
#        batch_y  = batch_y.cuda() # 推送到GPU
        out = net(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))
```
如果你的电脑有N卡并且显存足够大，就可以轻松把上面训练转移到GPU上跑，上面代码注释部分去掉即可。
![train_acc](train_acc.jpg)		

上面代码设计的batch_size为40，所有的样本数可能并非能被40整除，故训练一个epoch完还有些样本达不到40。简单迭代训练后，训练精度为0.98左右。		
		
### 模型保存和导出
---
为了把训练好的模型以后再使用或者在其他框架中使用，就可以导成onnx模式。代码如下：
```python
#%% 保存模型到磁盘
#参考：https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb
#torch.save(net.state_dict(),'judgeball.pkl')
torch.save(net,'myclassifyer.pkl')
dummy_input = Variable(torch.randn(1, *input_size))
torch.onnx.export(net, dummy_input, "myclassify.onnx")
```

## 测试网络
---
训练保存完模型后，现在就可测试网络的效果啦~注意测试脚本中，要写上网络结构和图像输入大小，而且要与训练脚本‘train_save.py’中设计的网络要一致，否则加载网络后会报错（踩坑过( ▼-▼ )）。
测试网络脚本为`main_test.py`
```python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:03:29 2018

@author: Cuixingxing
"""
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import os

#%% 网络结构
class testNet(nn.Module):
    def __init__(self,input_size=(3,300,300)):
        super(testNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        
        n_size = self._get_linear_inNums(input_size)#根据输入图像大小自动推断fc层输入
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(n_size , 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),        
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(), 
            torch.nn.Linear(10, 5)
            )
        
    def _get_linear_inNums(self,shape):
        batch_x = 1
        temp = Variable(torch.rand(batch_x,*shape))
        single_feature = self._forward_features(temp)
        n_size = single_feature.view(batch_x,-1).size(1)
        return n_size
    
    def _forward_features(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        out = self._forward_features(x)
        print('out.size():',out.size())
        res = out.view(out.size(0),-1)
        out = self.dense(res)
        out =  F.log_softmax(out,dim = 1)
        return out
    

net = torch.load(r'E:\python_work\pytorch_classify\myclassifyer.pkl')
transform1 = transforms.Compose([transforms.Resize((300,300)),
                                 transforms.ToTensor()]) # 或者直接image = transforms.ToTensor()(img)

rootdir = r'F:\imagesData\svm_images\test_image'
imageLists = os.listdir(rootdir)   
for imgname in imageLists: 
    img = Image.open(os.path.join(rootdir,imgname)).convert('RGB')#读入任意图像
    #Tensor转成PIL.Image重新显示
    #new_img_PIL = transforms.ToPILImage()(transform1(img)).convert('RGB')
    #new_img_PIL.show() # 处理后的PIL图片
    
    image = transform1(img).unsqueeze(0)
    #image = image.cuda() #如果在GPU上训练用这个，网络也要推送到GPU上，下面2句或可选
    #device = torch.device(0)
    #image = image.requires_grad_().to(device)
    
    net.eval()
    net.cpu()
    out = net(image) # image必须为n*c*h*w四维图像
    pred = torch.max(out, 1)[1] # 找到概率最大值对应的索引
    if pred==0:
        predLabel = 'airplane'
    elif pred==1:
        predLabel = 'butterfly'
    elif pred ==2:
        predLabel = 'camera'
    elif pred==3:
        predLabel ='scissors'
    else:
        predLabel = 'sunflower'
    
    img_numpy = image.squeeze().numpy()
    img_numpy = img_numpy.transpose((1,2,0))
    img_numpy = np.floor(img_numpy*255).astype('uint8')
    plt.figure()
    plt.imshow(img_numpy)
    plt.title('predLabel:'+predLabel)
    plt.show()
```
由于画面有限，截取部分预测图像，16张图像基本全部预测成功。
![1](1.jpg)
![2](2.jpg)
![3](3.jpg)
![4](4.jpg)
![5](5.jpg)
![6](6.jpg)

# Have fun ^_^ 
