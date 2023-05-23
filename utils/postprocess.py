from PIL import Image
import os.path
import glob
import numpy as np
import cv2


def resize(path1, path2):
    img = Image.open(path1)
    new_img = img.resize((1024, 768))
    new_img = new_img.convert("RGB")
    new_img.save(path2)

def resize_cv(path1, path2):
    img = cv2.imread(path1, 0)
    new_img = cv2.resize(img, (1024, 768))
    ans = np.zeros([768, 1024, 3])
    ans[:, :, 0] = new_img
    ans[:, :, 1] = new_img
    ans[:, :, 2] = new_img
    cv2.imwrite(path2, ans)
