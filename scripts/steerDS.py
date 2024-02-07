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
        original_filenames = glob(path.join(self.root_folder, "*" + self.img_ext))
        self.filenames = []
        self.steering_values = []
        
        # Process each file to determine if it should be doubled
        for f in original_filenames:
            steering = float(f.split("/")[-1].split(self.img_ext)[0][6:])
            self.filenames.append(f)
            self.steering_values.append(steering)
            # Double the entry if steering is not zero
            if steering != 0:
                self.filenames.append(f)
                self.steering_values.append(-steering)  # Use the negated steering for the augmented image
            

        # self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        # # Ensure each image is considered twice
        # self.filenames *= 2
        
        self.totensor = transforms.ToTensor()

        self.train_cut_folder = path.join(self.root_folder, "train_cut")
        # Create the train_cut folder if it doesn't exist
        if not path.exists(self.train_cut_folder):
            makedirs(self.train_cut_folder)
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]
        steering = self.steering_values[idx]        
        img = cv2.imread(f)
        cutter = CutImage(img)
        img = cutter.cutimage() # cut the top half of the image

        # Perform augmentation if the steering is inverted (i.e., it's a doubled entry)
        if steering != float(f.split("/")[-1].split(self.img_ext)[0][6:]):
            img = cv2.flip(img, 1)  # Flip horizontally

        # Define the path for the cut image
        cut_img_filename = path.join(self.train_cut_folder, path.basename(f))
        # Save the cut image
        cv2.imwrite(cut_img_filename, img)


        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   

        
        # todo: if steering is not zero, left-right inverse the img, and change the sign in front of the steering value. Add the modified steering-img data to the dataset
                      
        return img, steering
    

    # def cutimage(self, img):
    #     # cut the top half of the image
    #     height = img.shape[0]
    #     return img[height//2:,:]



        # # Extract steering value
        # steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        # steering = np.float32(steering)

        # # If steering is not zero, flip the image and invert the steering sign
        # if steering != 0:
        #     img = cv2.flip(img, 1)  # 1 indicates a horizontal flip
        #     steering = -steering  # Invert the steering sign
