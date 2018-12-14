#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/12/6 14:06
# @Author : Qiangz
# @File : video_process.py

import cv2
import dlib
import openface.align_dlib as ad
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

IMG_DIM = 96  # The edge length in pixels of the square the image is resized to.
IMAGE_TEST_PATH = "/home/public/MovieFaceTest/test_image"
VIDEO_TEST_PATH = "/home/public/MovieFaceTest/test_video"
FacePredictor = "/home/zq/openface/models/dlib/shape_predictor_68_face_landmarks.dat"

"""


"""
class VideoProcess(object):
    def __init__(self, face_predictor):
        self.AD = ad.AlignDlib(face_predictor)

    def video2images(self, video):
        assert video is not None

        images = []
        return images

    def image2landmarks(self, image):
        image = self.AD.align(imgDim=IMG_DIM, rgbImg=image)
        rec = dlib.rectangle()
        landmarks = self.AD.findLandmarks(image, bb=rec)
        return landmarks

    def images2list(self, images):
        faces_lm = []
        for image in images:
            faces_lm.append(self.image2landmarks(image))
        return faces_lm


vp = VideoProcess(FacePredictor)

images_names = os.listdir(IMAGE_TEST_PATH)
print(images_names)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FacePredictor)

for i, image_name in enumerate(images_names):
    image_path = IMAGE_TEST_PATH+"/"+image_name
    print(image_path)
    im = mpimg.imread(image_path)
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    print(im.shape)
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            print(idx, pos)

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(im, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", im)
    # cv2.waitKey(0)

    print("Image", i)
    lm = vp.image2landmarks(im)
    print("Landmarks:", i)
    print(lm)
    # im.show()