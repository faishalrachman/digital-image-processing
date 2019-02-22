# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:17:39 2018

@author: Faishal Rachman
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageUtils import greyscale
fname = "C:/Users/Faishal Rachman/Downloads/images.jpg"
img = cv2.imread(fname)
img_new = greyscale(img)
temp = np.zeros_like(img)

print(img_new[0,0])


def menyebarUpil(x,y,color, seed):
    try:
        cek = img_new[y,x,0]
        cek2 = temp[y,x,0]
        print(cek, color)
        if (cek - color < seed and cek2 == 0):
            temp[y,x] = [255,255,255]
            #JEJERAN ATAS
            menyebarUpil(x-1,y-1,color,seed)
            menyebarUpil(x,y-1,color,seed)
            menyebarUpil(x+1,y-1,color,seed)
            #JEJERAN TENGAH
            menyebarUpil(x-1,y,color,seed)
            menyebarUpil(x+1,y,color,seed)
            #JEJERAN Bawah
            menyebarUpil(x-1,y+1,color,seed)
            menyebarUpil(x,y+1,color,seed)
            menyebarUpil(x+1,y+1,color,seed)
    except:
        print("Batas")
    
menyebarUpil(155,39,img_new[39,155,0],5)

plt.imshow(temp)