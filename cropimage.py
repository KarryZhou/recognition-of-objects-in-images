# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:26:40 2018
@author: karry
"""

import os
import cv2
import numpy as np

num = 0

def CropImage(file_path,dest_path):
    global num
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=1)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    #cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)


    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = image[y1:y1+hight, x1:x1+width]
    #res = cv2.resize(cropImg, (1280,960), interpolation=cv2.INTER_AREA)
    #cv2.imshow("Image", cropImg)
    cv2.imwrite(dest_path, cropImg)
    print(num)
    num+=1
    cv2.waitKey(0)

# 遍历指定目录，显示目录下的所有文件名
def CropImage4File(filepath, destpath):
    pathDir = os.listdir(filepath)  # list all the path or file  in filepath
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        dest = os.path.join(destpath, allDir)
        if os.path.isfile(child):
            CropImage(child,dest)

if __name__ == '__main__':
    filepath = '.\m'  # source images
    destpath = '.\s'  # resized images saved here
    CropImage4File(filepath, destpath)
