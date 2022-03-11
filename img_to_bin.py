#!/usr/bin/python

import numpy as np
import os
import sys
from PIL import Image

rootdir = sys.argv[1]
k = 0

for imgdir in os.listdir("./" + rootdir):
    k += 1

rgb_arr = np.zeros([k, 256, 256, 3], dtype="int32")
gray_arr = np.zeros([k, 256, 256], dtype="int32")

k = 0
for imgdir in os.listdir("./" + rootdir):
    img = Image.open("./" + rootdir + "/" + imgdir)
    data = np.asarray(img.convert('RGB'))
    rgb_arr[k] = data
    gray_arr[k] = np.asarray(img.convert('L'))
    k += 1

np.save(rootdir + "_rgb", rgb_arr)
np.save(rootdir + "_gray", gray_arr)
