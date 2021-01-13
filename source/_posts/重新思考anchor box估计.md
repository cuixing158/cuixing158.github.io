---
title: 重新思考anchor box估计
catalog: true
date: 2020-12-30 12:25:53
subtitle: 
header-img: "photo.jpg"
tags: 
- kmeans
- anchorBox
catagories: 
- python
- 算法
---

> anchorBox估计是目标检测中非常经典的一个概念，首次在fasterRCNN中被提出，后续目标检测器如（SSD,YOLOv2/v3/v4）等都会借用这么一个好用的东西，其优势在于能提供更快的检测速度（一次计算图像中所有的预测）和解决目标多尺度问题。anchor可以翻译成“锚”，是能与实际的检测框最相似的东东，但如何得到这个“最相似”呢？下面本博客会逐渐介绍kmeans算法估计anchorbox.

&#160; &#160; &#160; &#160;众所周知，anchor是一组预先定义的宽高组成的boundingBox,如何设计才能让最相似的宽高anchor与真值尽可能接近呢？如果仅简单思考，很多人会认为取遍所有真值boundingBox的宽高进行聚类即可，虽说此方法可行，但并非相似上的最优。因为聚类距离采用的是宽高像素之间的欧式距离，不是boundingBox形状上的最优。所以原作者想到的是boundingBox之间的IOU ratio值作为聚类距离，值越大代表重合的比例越大，越相似。通常情况下，相似距离越小代表越相似，故衡量boundingBox的距离公式为1-IOU（boxA,boxB）,下面逐步介绍kmeans（kmeans++初始化）如何估计anchors,最后给出具体实现代码进行测试.

## kmeans++
kmeans算法是一种简单的无监督学习算法，根据事先设定的K个簇类中心和已有数据，多次迭代寻找最相似的簇类中心。由于其简单，其算法步骤略，主要介绍下kmeans++是如何初始化种子点的，其选择对收敛速度和全局最优解有影响。kmeans++的基本思想是：初始的聚类中心之间的相对距离尽可能远。算法步骤为：<br>

**1**、从数据集X中随机选择一个样本作为第一个聚类的中心(centroid)，记为c1;<br>
**2**、对数据集X中其他的每个样本Xm与c1进行IOU距离计算，样本m与已有簇类中心cj的距离标记为：d(xm,cj)；<br>
**3**、用以下列概率从X中选择另一个样本作为中心，记为c2；<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=p_{m}=\frac{d^{2}(x_{m},c_{1})}{\sum_{j}^{n}d^{2}(x_{j},c_{1})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{m}=\frac{d^{2}(x_{m},c_{1})}{\sum_{j}^{n}d^{2}(x_{j},c_{1})}" title="p_{m}=\frac{d^{2}(x_{m},c_{1})}{\sum_{j}^{n}d^{2}(x_{j},c_{1})}" /></a>
**4**、选择下一个簇类中心cj,先计算每个样本与已有的簇类中心的距离，找到每个样本归属的簇类中心。然后根据每个样本m=1,2,...,n，p=1,2,...,j-1,从X中以
<a href="https://www.codecogs.com/eqnedit.php?latex=p=\frac{d^{2}(x_{m},C_{p})}{\sum_{x_{h}\subseteq&space;C_{p}}^{}d^{2}(x_{h},C_{p})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=\frac{d^{2}(x_{m},C_{p})}{\sum_{x_{h}\subseteq&space;C_{p}}^{}d^{2}(x_{h},C_{p})}" title="p=\frac{d^{2}(x_{m},C_{p})}{\sum_{x_{h}\subseteq C_{p}}^{}d^{2}(x_{h},C_{p})}" /></a>
概率选择下一个初始中心。其中，Cp是所有样本到离最近的簇类中心centroid 的集合,即选择下一个样本点作为簇类中心的概率是正比于其归属于已有簇类中心距离；<br>
**5**、重复步骤4，直到有K个centroids被选择。

## code
下面给出具体python代码实现：<br>
```python
def bboxesOverRation(bboxesA,bboxesB):
    """
    功能等同于matlab的函数bboxesOverRation
    bboxesA：M*4 array,形如[x,y,w,h]排布
    bboxesB: N*4 array,形如[x,y,w,h]排布
    """
    bboxesA = np.array(bboxesA.astype('float'))
    bboxesB = np.array(bboxesB.astype('float'))
    M = bboxesA.shape[0]
    N = bboxesB.shape[0]
    
    areasA = bboxesA[:,2]*bboxesA[:,3]
    areasB = bboxesB[:,2]*bboxesB[:,3]
    
    xA = bboxesA[:,0]+bboxesA[:,2]
    yA = bboxesA[:,1]+bboxesA[:,3]
    xyA = np.stack([xA,yA]).transpose()
    xyxyA = np.concatenate((bboxesA[:,:2],xyA),axis=1)
    
    xB = bboxesB[:,0] +bboxesB[:,2]
    yB = bboxesB[:,1]+bboxesB[:,3]
    xyB = np.stack([xB,yB]).transpose()
    xyxyB = np.concatenate((bboxesB[:,:2],xyB),axis=1)
    
    iouRatio = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            x1 = max(xyxyA[i,0],xyxyB[j,0]);
            x2 = min(xyxyA[i,2],xyxyB[j,2]);
            y1 = max(xyxyA[i,1],xyxyB[j,1]);
            y2 = min(xyxyA[i,3],xyxyB[j,3]);
            Intersection = max(0,(x2-x1))*max(0,(y2-y1));
            Union = areasA[i]+areasB[j]-Intersection;
            iouRatio[i,j] = Intersection/Union; 
    return iouRatio
   
def estimateAnchorBoxes(trainingData,numAnchors=9):
    '''
    功能：kmeans++算法估计anchor，类似于matlab函数estimateAnchorBoxes,当trainingData
    数据量较大时候，自写的kmeans迭代循环效率较低，matlab的estimateAnchorBoxes得出
    anchors较快，但meanIOU较低，然后乘以实际box的ratio即可。此算法由于优化是局部，易陷入局部最优解，结果不一致属正常
    cuixingxing150@gmail.com
    Example: 
        import scipy.io as scipo
        data = scipo.loadmat(r'D:\Matlab_files\trainingData.mat')
        trainingData = data['temp']
        
        meanIoUList = []
        for numAnchor in np.arange(1,16):
            anchorBoxes,meanIoU = estimateAnchorBoxes(trainingData,numAnchors=numAnchor)
            meanIoUList.append(meanIoU)
        plt.plot(np.arange(1,16),meanIoUList,'ro-')
        plt.ylabel("Mean IoU")
        plt.xlabel("Number of Anchors")
        plt.title("Number of Anchors vs. Mean IoU")
        
    Parameters
    ----------
    trainingData : numpy 类型
        形如[x,y,w,h]排布，M*4大小二维矩阵
    numAnchors : int, optional
        估计的anchors数量. The default is 9.

    Returns
    -------
    anchorBoxes : numpy类型
        形如[w,h]排布，N*2大小矩阵.
    meanIoU : scalar 标量
        DESCRIPTION.
    
    '''
    
    numsObver = trainingData.shape[0]
    xyArray = np.zeros((numsObver,2))
    trainingData[:,0:2] = xyArray
    assert(numsObver>=numAnchors)
    
    # kmeans++
    # init 
    centroids = [] # 初始化中心，kmeans++
    centroid_index = np.random.choice(numsObver, 1)
    centroids.append(trainingData[centroid_index])
    while len(centroids)<numAnchors:
        minDistList = []
        for box in trainingData:
            box = box.reshape((-1,4))
            minDist = 1
            for centroid in centroids:
                centroid = centroid.reshape((-1,4))
                ratio = (1-bboxesOverRation(box,centroid)).item()
                if ratio<minDist:
                    minDist = ratio
            minDistList.append(minDist)
            
        sumDist = np.sum(minDistList)
        prob = minDistList/sumDist 
        idx = np.random.choice(numsObver,1,replace=True,p=prob)
        centroids.append(trainingData[idx])
        
    # kmeans 迭代聚类
    maxIterTimes = 100
    iter_times = 0
    while True:
        minDistList = []
        minDistList_ind = []
        for box in trainingData:
            box = box.reshape((-1,4))
            minDist = 1
            box_belong = 0
            for i,centroid in enumerate(centroids):
                centroid = centroid.reshape((-1,4))
                ratio = (1-bboxesOverRation(box,centroid)).item()
                if ratio<minDist:
                    minDist = ratio
                    box_belong = i
            minDistList.append(minDist)
            minDistList_ind.append(box_belong)
        centroids_avg = []
        for _ in range(numAnchors):
            centroids_avg.append([])
        for i,anchor_id in enumerate(minDistList_ind):
            centroids_avg[anchor_id].append(trainingData[i])
        err = 0
        for i in range(numAnchors):
            if len(centroids_avg[i]):
                temp = np.mean(centroids_avg[i],axis=0)
                err +=  np.sqrt(np.sum(np.power(temp-centroids[i],2)))
                centroids[i] = np.mean(centroids_avg[i],axis=0)
        iter_times+=1
        if iter_times>maxIterTimes or err==0:
            break
    anchorBoxes = np.array([x[2:] for x in centroids])
    meanIoU = 1-np.mean(minDistList)
    return anchorBoxes,meanIoU
```

![anchors_iou](anchors_iou.jpg)

## Reference
[1] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242 )
[2] [Anchor Boxes for Object Detection](https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html )



