from utils.dataset_png import train_2Dpng_transforms, test_2Dpng_transforms
import os
from glob import glob
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset, DataLoader
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from pathlib import Path

class MRI_2Dpng_Dataset(Dataset):
    def __init__(self, data_dirpath, transform):

        self.transform = transform
        self.fnames = []
        self.patient_to_indices = {}  

        patient_dirs = sorted([d for d in glob(os.path.join(data_dirpath,"*")) 
                            if os.path.isdir(d)])
        
        if not patient_dirs:
            raise ValueError(f"未找到任何患者文件夹: {data_dirpath}")
        
        for patient_dir in patient_dirs:
            png_files = sorted(glob(os.path.join(patient_dir, "*.png")))
            if not png_files:
                print(f"警告: 患者文件夹无PNG文件: {patient_dir}")
                continue
            
            patient_id = Path(patient_dir).name
            start_idx = len(self.fnames)
            self.fnames.extend(png_files)
            self.patient_to_indices[patient_id] = list(range(start_idx, len(self.fnames)))
        
        if not self.fnames:
            raise ValueError(f"未找到任何PNG文件: {data_dirpath}")
        
        print(f"成功加载 {len(self.fnames)} 个PNG文件 (来自 {len(self.patient_to_indices)} 个患者)")

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        try:
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"加载失败 {img_path}: {str(e)}")

            return torch.zeros(3, 256, 256) if self.transform else Image.new('RGB', (256, 256))

    def __len__(self):
        return len(self.fnames)

class FoldDataLoader:
    def __init__(self, data_dir, batch_size=64, num_workers=4, random_state=728, n_splits=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.n_splits = n_splits
        
        self.full_dataset = MRI_2Dpng_Dataset(
            data_dirpath=self.data_dir,
            transform=None
        )
        
        if len(self.full_dataset) == 0:
            raise ValueError("No valid images found in the dataset")

        self._init_patient_level_splits()
        
    def _init_patient_level_splits(self):
        patient_ids = list(self.full_dataset.patient_to_indices.keys())

        self.train_patients, self.val_patients = train_test_split(
            patient_ids,
            test_size=0.2,
            random_state=self.random_state,
            shuffle=True
        )
        
        self.train_indices = []
        for pat in self.train_patients:
            self.train_indices.extend(self.full_dataset.patient_to_indices[pat])
            
        self.val_indices = []
        for pat in self.val_patients:
            self.val_indices.extend(self.full_dataset.patient_to_indices[pat])

        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.splits = list(self.kf.split(self.train_patients))  
    
    def get_fold(self, fold_num):

        fold_idx = fold_num - 1
        if fold_idx < 0 or fold_idx >= self.n_splits:
            raise ValueError(f"fold_num must be between 1 and {self.n_splits}")

        train_pat_idx, val_pat_idx = self.splits[fold_idx]
        fold_train_patients = [self.train_patients[i] for i in train_pat_idx]
        fold_val_patients = [self.train_patients[i] for i in val_pat_idx]

        fold_train_indices = []
        for pat in fold_train_patients:
            fold_train_indices.extend(self.full_dataset.patient_to_indices[pat])
            
        fold_val_indices = []
        for pat in fold_val_patients:
            fold_val_indices.extend(self.full_dataset.patient_to_indices[pat])
        
        print(f"\nLoading Fold {fold_num}")
        print(f"  训练患者: {len(fold_train_patients)} 个")
        print(f"  验证患者: {len(fold_val_patients)} 个")

        train_dataset = Subset(self.full_dataset, fold_train_indices)
        train_dataset.dataset.transform = train_2Dpng_transforms
        
        val_dataset = Subset(self.full_dataset, fold_val_indices)
        val_dataset.dataset.transform = test_2Dpng_transforms
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader
    
    def get_fixed_val_loader(self):

        test_dataset = Subset(self.full_dataset, self.val_indices)
        test_dataset.dataset.transform = test_2Dpng_transforms
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    def get_fixed_train_loader(self):

        train_dataset = Subset(self.full_dataset, self.train_indices)
        train_dataset.dataset.transform = train_2Dpng_transforms
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    def get_all_loader(self):

        all_indices = list(range(len(self.full_dataset)))
        

        all_dataset = Subset(self.full_dataset, all_indices)
        all_dataset.dataset.transform = test_2Dpng_transforms 
        
        return DataLoader(
            all_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_patient_splits_info(self):
        return {
            'train_patients': self.train_patients,
            'val_patients': self.val_patients,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices
        }
    
