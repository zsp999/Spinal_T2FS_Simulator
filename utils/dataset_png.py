import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from glob import glob
from PIL import Image


train_dir_png = "/your/train_data/path"
test_dir_png = "/your/test_data/path"

HW=256

train_2Dpng_transforms = T.Compose([ 
        T.RandomHorizontalFlip(), 
        T.RandomVerticalFlip(),

        T.RandomApply(transforms=[T.RandomResizedCrop(size=HW, scale=(0.25, 1), ratio=(3 / 4, 4 / 3), antialias=True)],p=0.5), 
        T.Resize(HW, antialias=True),
        T.CenterCrop(HW), 
        T.ToTensor(), 
        T.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)), 
        ])
test_2Dpng_transforms= T.Compose([ 
        T.Resize(HW, antialias=True),
        T.CenterCrop(HW),       
        T.ToTensor(), 
        T.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)), 
        ])


class MRI_2Dpng_Dataset(Dataset):
    def __init__(self, data_dirpath, transform, focal_str = '*'): 
        self.fnames = glob(os.path.join(data_dirpath, focal_str))
        self.transform = transform
    
    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx])
        img = self.transform(img)
        return img
 
    def __len__(self):
        return len(self.fnames)


