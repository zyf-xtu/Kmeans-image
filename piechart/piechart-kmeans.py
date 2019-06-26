#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-25 23:44
# @Author  : zhangyunfei
# @File    : piechart-kmeans.py
# @Software: PyCharm
'''
    对饼状图进行聚类分析，解析出每个颜色所占比重
    1.需要输入饼状图像
    2.需要事先知道饼状图有多少种类别
'''

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import cv2 as cv
import numpy as np
from numpy import delete


# 根据聚类的质心，确定直方图
def centroid_histogram(clt):
    # 抓取不同簇的数量并创建直方图
    # 基于分配给每个群集的像素数
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    clustercenters = clt.cluster_centers_
    count = 0
    label = -1
    for i in clustercenters:
        # print(i.tolist())
        center = i.tolist()
        if center[0] > 200 and center[1] > 200 and center[2] > 200:  #判断白色底色类簇中心点
            label = count
        count += 1

    if label > -1:
        labels = clt.labels_.tolist()
        labelsnew = []
        for i in labels:
            if i != label:
                labelsnew.append(i)           #重新生成一个出去白色底色的列表

        dd = numLabels.tolist()
        dd.remove(label)             #移除白色底色的聚类标签
        clt.labels_ = np.array(labelsnew)
        numLabels = np.array(dd)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    else:
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize直方图，使其总和为1
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist, label


def plot_colors(hist, centroids):
    # 初始化表示相对频率的条形图每种颜色的＃
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # 循环遍历每个类簇的百分比和颜色,每个集群
    for (percent, color) in zip(hist, centroids):
        # 绘制每个类簇的相对百分比
        print(percent, color)
        print(str(percent))
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                     color.astype("uint8").tolist(), -1)
        cv.putText(bar, str(round(percent * 100, 1)) + '%', (int(startX), 20), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4,
                   (0, 0, 0), 1)
        startX = endX

    return bar


if __name__ == '__main__':
    # 加载图像
    image_base = cv.imread("timg.jpg")
    # 初始化图像像素列表
    image = image_base.reshape((image_base.shape[0] * image_base.shape[1], 3))
    k = 7  # 聚类的类别个数+1,1为白色底色类别，每次都要重新定义
    iterations = 4  # 并发数4
    iteration = 300  # 聚类最大循环次数
    clt = KMeans(n_clusters=k, n_jobs=iterations, max_iter=iteration)#kmeans聚类
    clt.fit(image)
    hist, label = centroid_histogram(clt)
    print(hist, label)
    clusters = clt.cluster_centers_
    if label>-1:
        clt.cluster_centers_ = delete(clt.cluster_centers_, label, axis=0)
    bar = plot_colors(hist, clt.cluster_centers_)
    # 展示
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.imshow(image_base)
    ax = fig.add_subplot(212)
    ax.imshow(bar)
    plt.show()
    cv.waitKey(0)
