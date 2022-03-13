#!/usr/bin/python

import numpy as np
import os
import sys
from PIL import Image

rootdir = sys.argv[1]
k = 0

for imgdir in os.listdir("./" + rootdir):
    k += 1

img = Image.open("./" + rootdir + "/" + imgdir)
data = np.asarray(img.convert('RGB'), dtype="uint8")
imgDim = data.shape[0]

rgb_arr = np.zeros([k, imgDim, imgDim, 3], dtype="uint8")
gray_arr = np.zeros([k, imgDim, imgDim], dtype="uint8")

k = 0
for imgdir in os.listdir("./" + rootdir):
    img = Image.open("./" + rootdir + "/" + imgdir)
    data = np.asarray(img.convert('RGB'), dtype="uint8")
    rgb_arr[k] = data
    gray_arr[k] = np.asarray(img.convert('L'), dtype="uint8")
    k += 1

np.save(rootdir + "_rgb", rgb_arr)
np.save(rootdir + "_gray", gray_arr)
