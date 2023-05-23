import os
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np


def gamma_correction(img_origin, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img_origin, lookUpTable)


def mean_correction(img_ori, mean_delta):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(i - mean_delta, 0, 255)
    return cv2.LUT(img_ori, lookUpTable)
