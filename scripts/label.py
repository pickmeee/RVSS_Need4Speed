from scripts.steerDS import SteerDataSet
import os
from os import path
import glob
import numpy as np


def label(filename, img_ext = ".jpg"):
    steering = filename.split("/")[-1].split(img_ext)[0][6:]
    img_id = f.split("/")[-1].split(img_ext)[0][:6]
    print(img_id)
    steering = np.float32(steering)
    if steering < 0:
        return "LEFT"
    elif steering > 0:
        return "RIGHT"
    else:
        return "STRAIGHT"


script_path = os.path.dirname(os.path.realpath(__file__))
ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train', 'raw_data'), '.jpg')
print(ds.__len__())

for f in ds.filenames:
    label(f)