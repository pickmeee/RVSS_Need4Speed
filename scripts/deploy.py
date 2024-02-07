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
from PIL import Image

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
PATH = os.path.join(script_path, '..', 'models/model_1.pth')
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
        im = Image.fromarray(im.transpose(1, 2, 0))  # Transpose the dimensions from (C, H, W) to (H, W, C)

        #TO DO: apply any necessary image transforms
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cutter = CutImage(im)
        im = cutter.cutimage()
        im = transform(im)

        # Apply any necessary image transforms (assuming these expect a PIL image)
        im = transform(im)

        # At this point, 'im_transformed' is a tensor of shape (C, H, W)
        # Add a batch dimension to make it (N, C, H, W) where N=1
        im = im.unsqueeze(0)

        # print()

        
        
        print(f'shape: {im.shape}')

        # torch.reshape(im, (1, 3,120,320))

        #TO DO: pass image through network get a prediction
        outputs = net(im)
        _, predicted = torch.max(outputs.data, 1)

        #TO DO: convert prediction into a meaningful steering angle
        if predicted.numpy()[0] == 0:
            angle -= 0.3
        elif predicted.numpy()[0] == 1:
            angle = 0
        else:
            angle += 0.3

        #TO DO: check for stop signs?
        
        # angle = 0

        Kd = 10 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 25 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
