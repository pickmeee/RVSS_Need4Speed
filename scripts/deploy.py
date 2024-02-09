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
from machinevisiontoolbox import Image

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE
net = Net()

#LOAD NETWORK WEIGHTS HERE
script_path = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(script_path, '..', 'models/train_steer_class_net_modified.pth')
net.load_state_dict(torch.load(PATH))

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()

        # Convert the NumPy array to a PIL image if necessary (assuming 'im' is RGB format)
        # im = Image.fromarray(im.transpose(1, 2, 0))  # Transpose the dimensions from (C, H, W) to (H, W, C)

        
        
        #TO DO: apply any necessary image transforms
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ColorJitter(contrast=0.15, saturation=1),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cutter = CutImage(im)
        im = cutter.cutimage()

        #stop sign #####################
        kernel = np.ones((9,9),np.float32)/81
        image = cv2.filter2D(im,-1,kernel)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define a range for the red color in HSV
        lower_red = np.array([0, 120, 80])
        upper_red = np.array([10, 255, 255])
        # Create a mask using the inRange function
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # Bitwise AND the original image with the mask
        result = cv2.bitwise_and(image, image, mask=mask)

        gray_img=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        blobs = thresh.blobs()
        if len(blobs) > 0:
            for blob in range(len(blobs)):
                if blobs[blob].area > 100:
                    print("Stop Sign!")
                    bot.setVelocity(0, 0)
                    time.sleep(0.5)
                    break

        ################


        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        saturation_enhancer = ImageEnhance.Color(im)
        im = saturation_enhancer.enhance(factor = 2)

        contrast_enhancer = ImageEnhance.Contrast(im)
        im = contrast_enhancer.enhance(factor = 1.5)


        # im = transform(im)

        # Apply any necessary image transforms (assuming these expect a PIL image)
        im = transform(im)

        # At this point, 'im_transformed' is a tensor of shape (C, H, W)
        # Add a batch dimension to make it (N, C, H, W) where N=1
        im = im.unsqueeze(0)

        # print()

        
        
        # print(f'shape: {im.shape}')

        # torch.reshape(im, (1, 3,120,320))

        
        #TO DO: pass image through network get a prediction
        outputs = net(im)
        _, predicted = torch.max(outputs.data, 1)



        #TO DO: convert prediction into a meaningful steering angle
        if predicted.numpy()[0] == 0:
            angle = -0.2
        elif predicted.numpy()[0] == 1:
            angle = 0
        else:
            angle = +0.2
        
        print(predicted.numpy()[0])

        #TO DO: check for stop signs?
        
        # angle = 0

        Kd = 10 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 25 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
