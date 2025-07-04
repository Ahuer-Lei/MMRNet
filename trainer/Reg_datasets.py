import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import setup_seed
from utils.deformation import affine,non_affine_2d
import torch.nn.functional as F
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
        self.non_affine = non_affine_2d
        self.affine = affine

    def __getitem__(self, index):

        item_opt = imread(self.files_opt[index % len(self.files_opt)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_opt_1, gt_tp, _ = self.affine(random_numbers=random_numbers, imgs=[item_opt], padding_modes=['zeros'], opt=self.opt)
        item_sar_1, gt_tp, _ = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_sar = item_sar.squeeze(0)
        item_opt = item_opt.squeeze(0)
           
        return item_opt, item_sar, item_opt_1, item_sar_1, gt_tp

    def __len__(self):
        return len(self.files_opt)


class TestDataset(Dataset):
    def __init__(self, root, opt):
         
        self.files_opt = sorted(glob.glob("%s/opt/*" % root))
        self.files_sar = sorted(glob.glob("%s/sar/*" % root))

        self.opt = opt
        self.non_affine = non_affine_2d
        self.affine = affine

    def __getitem__(self, index):
  
        item_opt = imread(self.files_opt[index % len(self.files_opt)])
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_opt_1, gt_tp, four_point = self.affine(random_numbers=random_numbers, imgs=[item_opt], padding_modes=['zeros'], opt=self.opt)
        item_sar_1, gt_tp, four_point = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_sar = item_sar.squeeze(0)
        item_opt = item_opt.squeeze(0)
           
        return item_opt, item_sar, item_opt_1, item_sar_1, gt_tp, four_point

    def __len__(self):
        return len(self.files_opt)



  
# class InferDataset(Dataset):
#     def __init__(self, root, transforms_, opt):
#         self.transforms = transforms.Compose(transforms_)
        
#         self.files_opt = sorted(glob.glob("%s/optnpy/*" % root))
#         self.files_sar = sorted(glob.glob("%s/sarnpy/*" % root))
#         self.files_gtp = sorted(glob.glob("%s/gt_tp/*" % root))

#         self.opt = opt
#         self.non_affine = non_affine_2d
#         self.affine = affine

#     def __getitem__(self, index):

#         name = self.files_opt[index % len(self.files_opt)].split('/')[-1].split('.')[0]
#         item_opt = self.transforms(np.load(self.files_opt[index % len(self.files_opt)]).astype(np.float32))   # RA
#         item_sar = self.transforms(np.load(self.files_sar[index % len(self.files_sar)]).astype(np.float32))  # RB

#         gt_tp = np.load(self.files_gtp[index % len(self.files_gtp)]).astype(np.float32)       # RB
#         a = np.array([[0,0,1]])
#         theta = np.concatenate([gt_tp, a])
#         matrix = np.linalg.inv(theta)

#         matrix = matrix[:-1, :]
#         theta = torch.from_numpy(matrix).to(torch.float32)
#         grid = F.affine_grid(theta.unsqueeze(0), [1,1,256,256], align_corners=True)
#         sar_warp = F.grid_sample(item_sar, grid, align_corners=True, padding_mode='zeros').squeeze(0)


#         item_opt = item_opt.squeeze(0)
           
#         return item_opt, sar_warp, name

#     def __len__(self):
#         return len(self.files_opt)  
    
    
