from glob import glob
from os import path
from torchvision import transforms

class CutImage():
    def __init__(self,img=None):
        self.img = img
    
    def cutimage(self):
        # cut the top half of the image
        height = self.img.shape[0]
        return self.img[height//2:,:]

    