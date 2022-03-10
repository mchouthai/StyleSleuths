#!/usr/bin/python

from http.server import ThreadingHTTPServer
import numpy as np
import os
import sys
import PIL
from PIL import Image
import math


rootdir = sys.argv[1]

robertsX = np.array([[1, 0], [0, -1]])
robertsY = np.array([[0, 1], [-1, 0]])
sobelX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
sobelY = np.rot90(sobelX)
basic = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

rgb_arr = np.load(rootdir + "_rgb.npy")
gray_arr = np.load(rootdir + "_gray.npy")
k = rgb_arr.shape[0]


def sum_imgs(edges=False):
    rgb_sums = np.zeros([k, 3], dtype="float32")
    gray_sums = np.zeros([k, 1], dtype="int32")
    edge_sums = np.zeros([k, 1], dtype="int32")
    for i in range(0, k):
        gray_sums[i] = np.sum(gray_arr[i], (0, 1))
        rgb_sums[i] = np.sum(rgb_arr[i], (0, 1)) / gray_sums[i]
        if(edges):
            edge_sums[i] = np.sum(
                convolve(gray_arr[i], sobelX, sobelY), (0, 1))
        print("Processing img " + str(i))
    return rgb_sums, gray_sums, edge_sums


def convolve(image, kernel, kernelY=None, threshold=150):

    imgY = image.shape[0]
    imgX = image.shape[1]

    kSize = kernel.shape[0]

    image_padded = np.zeros((imgY + kSize, imgX + kSize), dtype="int8")
    image_padded[0:imgY, 0:imgX] = image
    out = np.zeros((imgY, imgX))

    for y in range(0, imgY):
        if y > imgY:
            break
        for x in range(0, imgX):
            if x > imgX:
                break
            if kernelY is not None:
                Gx = np.sum(
                    kernel * image_padded[x: x + kSize, y: y + kSize])
                Gy = np.sum(
                    kernelY * image_padded[x: x + kSize, y: y + kSize])
                G = abs(Gx) + abs(Gy)
            else:
                G = np.sum(
                    kernel * image_padded[x: x + kSize, y: y + kSize])
            out[x, y] = G if (abs(G) > threshold) else 0
    return out


def save(edges):
    rgb_sums, gray_sums, edge_sums = sum_imgs(edges)
    np.save(rootdir + "_rgbsum", rgb_sums)
    np.save(rootdir + "_graysum", gray_sums)
    if (edges):
        np.save(rootdir + "_edgesum", edge_sums)
    return


save(False)
