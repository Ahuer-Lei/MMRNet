import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import utils.deformation as deformation
from utils.deformation import affine
import cv2
import kornia.utils as KU

def imread(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_ts = ((KU.image_to_tensor(img) / 127.5) -1).float()
    im_ts = im_ts.unsqueeze(0)
    return im_ts

class ImageDataset(Dataset):
    def __init__(self, root, opt):
        self.files_opt = sorted(glob.glob("%s/opt/*" % root))
        self.files_sar = sorted(glob.glob("%s/sar/*" % root))
        self.opt = opt
        self.affine = affine

    def __getitem__(self, index):
        
        item_opt = imread(self.files_opt[index % len(self.files_opt)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        # make same spatial affine transform  opt and sar 
 
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_opt_1 = self.affine(random_numbers=random_numbers, imgs=[item_opt], padding_modes=['zeros'], opt=self.opt)
        item_sar_1 = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_opt_2 = item_opt.squeeze(0)
     
        label = (item_opt_1 + 1) / 2 - (item_opt_2 + 1) / 2   # make (opt-sar ) as label and keep range (-1,1)

        return item_opt_2, item_opt_1, item_sar_1, label

    def __len__(self):
        return len(self.files_opt)
    

    
class TestDataset(Dataset):
    def __init__(self, root, opt):
        self.files_opt = sorted(glob.glob("%s/opt/*" % root))
        self.files_sar = sorted(glob.glob("%s/sar/*" % root))
        self.opt = opt
        self.affine = affine

    def __getitem__(self, index):
       
        item_opt = imread(self.files_opt[index % len(self.files_opt)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        # make same spatial affine transform  opt and sar 
  
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_opt_1 = self.affine(random_numbers=random_numbers, imgs=[item_opt], padding_modes=['zeros'], opt=self.opt)
        item_sar_1 = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_opt_2 = item_opt.squeeze(0)
     
        label = (item_opt_1 + 1) / 2 - (item_opt_2 + 1) / 2   # make (opt-sar ) as label and keep range (-1,1)

        return item_opt_2, item_sar_1, label


    def __len__(self):
        return len(self.files_opt)   
    