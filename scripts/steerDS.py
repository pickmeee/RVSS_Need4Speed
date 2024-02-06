import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path, makedirs

from cutimg import CutImage

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()

        self.train_cut_folder = path.join(self.root_folder, "train_cut")
        # Create the train_cut folder if it doesn't exist
        if not path.exists(self.train_cut_folder):
            makedirs(self.train_cut_folder)
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)
        cutter = CutImage(img)
        img = cutter.cutimage() # cut the top half of the image

        # Define the path for the cut image
        cut_img_filename = path.join(self.train_cut_folder, path.basename(f))
        # Save the cut image
        cv2.imwrite(cut_img_filename, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(steering)

        if steering < 0:
            label = "LEFT"
            label_id = 0
        elif steering > 0:
            label = "RIGHT"
            label_id = 1
        else:
            label = "STRAIGHT"
            label_id = 2
                      
        return img, steering, label_id

    # def cutimage(self, img):
    #     # cut the top half of the image
    #     height = img.shape[0]
    #     return img[height//2:,:]
