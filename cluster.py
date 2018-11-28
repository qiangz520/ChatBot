#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/26 11:30
# @Author : Qiangz
# @File : cluster.py

from sklearn.cluster import KMeans
from helpers import *
from parameters import *
import numpy as np

filenames = get_filename(TEST_PATH, "json")
print("Reading JsonFiles! Please wait...")
allData = []
for filename in filenames:
    text = loadfile_json(filename)
    for temp in text:
        if len(temp['text'].split(' ')) < MAX_LENGTH and \
                len(temp['text_next'].split(' ')) < MAX_LENGTH and \
                temp['facs_next_exist'] == 1 and len(temp['facs']) < MAX_LENGTH and \
                len(temp['facs_next']) < MAX_LENGTH:
            allData.append(temp)


"""
facesCluster Class
User kmeans algorithm to obtain gesture templates
Initialize kmeans

all_faces:List;Include all faces
kmeans:the obtained estimator by using sklearn.KMeans
getFacesIndexs:predict certain faces's cluster index, return a list

Create:@Qiangz @2018.11.26
Update:
"""


class facesCluster(object):
    def __init__(self):

        all_faces = []
        for temp in allData:
            if temp["facs"]:
                for temp_face in temp["facs"]:
                    all_faces.append(temp_face)
            if temp["facs_prev"]:
                for temp_face in temp["facs_prev"]:
                    all_faces.append(temp_face)
        self.all_faces =np.array(all_faces)
        self.kmeans = KMeans(n_clusters=10,random_state=0).fit(self.all_faces)

    def get_faces_indexs(self,faces):
        return self.kmeans.predict(faces)

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_


faces_cluster = facesCluster()

testFaces = faces_cluster.get_faces_indexs(allData[1]["facs"])
cluster_centers = faces_cluster.get_cluster_centers()
print("cluster_centers:")
print(cluster_centers)
print("testData:")
print(allData[1]["facs"])
print("testDataIndex:")
print(testFaces)
