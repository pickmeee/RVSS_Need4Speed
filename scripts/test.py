#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
from structure_net import Net
from cutimg import CutImage
from PIL import Image, ImageEnhance
import copy

im2 = cv2.imread("/home/yanzhang/rvss_ws/RVSS_Need4Speed/data/0000090.00.jpg")
kernel = np.ones((9,9),np.float32)/81
image = cv2.filter2D(im2,-1,kernel)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a range for the red color in HSV
lower_red = np.array([100, 100, 100])
upper_red = np.array([190, 255, 255])
# Create a mask using the inRange function
mask = cv2.inRange(hsv, lower_red, upper_red)
# Bitwise AND the original image with the mask
result = cv2.bitwise_and(image, image, mask=mask)

gray_img=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

detector = cv2.SimpleBlobDetector_create()

cv2.imwrite("ht.jpg", thresh)
print(thresh)

blobs = detector.detect(thresh)


if len(blobs) > 0:
    for blob in range(len(blobs)):
        if blobs[blob].area > 36:
            print("Stop Sign!")
            # bot.setVelocity(0, 0)
            time.sleep(0.5)
            break