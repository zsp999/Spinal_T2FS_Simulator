
import sys 
sys.path.append("..") 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
os.environ["WORLD_SIZE"] = "1"

from model.MMGSN.syn_model_2D_T2FS import *

import torch
import torch.nn as nn
from torch import optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR
from torch.cuda.amp import autocast
from torch.amp import  GradScaler
from utils.FolderDataLoader import *
from utils.test_metrics_2D import *

lr = 4e-5 
epochs = 500


device = 'cuda:0'
model_img = Multi_modal_generator().to(device)
discri_model = Discriminator().to(device)
model_img.apply(weights_init_normal)
discri_model.apply(weights_init_normal)



data_loader = FoldDataLoader(
    data_dir="/your/path",
    batch_size=64,
    num_workers=8,
    random_state=728,
    n_splits=5
)
fold_num = 'All'

train_loader = data_loader.get_fixed_train_loader()  
val_loader = data_loader.get_fixed_val_loader()     


G_optimizer = optim.AdamW(model_img.parameters(), lr=lr) 
D_optimizer = optim.AdamW(discri_model.parameters(), lr=lr)


G_scheduler = CosineAnnealingWarmRestarts(G_optimizer, T_0=5, T_mult=1, eta_min=1e-6) 
D_scheduler = CosineAnnealingWarmRestarts(D_optimizer, T_0=5, T_mult=1, eta_min=1e-6) 


criterion_GAN   = nn.MSELoss()
criterion_identity = nn.L1Loss()

scaler = GradScaler()
save_name = f"/your/path/log/fold{fold_num}"
os.makedirs(save_name, exist_ok=True)
changes = "abcd"
log_file = open(f"{save_name}/loss_metrics_{changes}.csv", 'a+', buffering=1)

log_file.write("time,epoch,iter,G_lr,loss_G,loss_D,SSIM,PSNR,MSE,AUC\n") 

ckpt_path = f"/your/path/model_save/fold{fold_num}.pt"

def save(model_img, epoch,  savename):
    checkpoint = {
        'epoch': epoch,
        'model': model_img,
    }
    torch.save(checkpoint, savename)

for epoch in range(epochs):  

        log_file.write(f"{get_time_now()},{epoch}, , , , , , , , \n") 
        model_img.train()
        for data_iter_step, data_batch in enumerate(train_loader):
                
                img_data, label_data = data_batch[:, :2, :, :], data_batch[:, 2:3, :, :]
                realpatch_label, fakepatch_label = torch.ones((label_data.shape[0],1,14,14), dtype=torch.float32, device=device), \
                        torch.zeros((label_data.shape[0],1,14,14), dtype=torch.float32, device=device)

                G_optimizer.zero_grad() 
                with autocast():
                        fake_img =  model_img(img_data.to(device))
                        loss_recon = criterion_identity(label_data.to(device), fake_img)
                        loss_gan = criterion_GAN(discri_model(fake_img), realpatch_label)
                        loss_G = (loss_gan + loss_recon)*10
                
                scaler.scale(loss_G).backward()
                scaler.unscale_(G_optimizer)
                scaler.step(G_optimizer)
                scaler.update()

                D_optimizer.zero_grad()
                with autocast():
                        loss_real = criterion_GAN(discri_model(label_data.to(device)), realpatch_label)
                        loss_fake = criterion_GAN(discri_model(fake_img.detach()), fakepatch_label) 
                        loss_D = loss_real + loss_fake
                
                scaler.scale(loss_D).backward()
                scaler.unscale_(D_optimizer)
                scaler.step(D_optimizer)
                scaler.update()
                
                G_scheduler.step(epoch+data_iter_step/len(train_loader))
                D_scheduler.step(epoch+data_iter_step/len(train_loader))

 
                if data_iter_step %5 == 0:
                        log_file.write(f"{get_time_now()},{epoch},{data_iter_step},{G_scheduler.get_last_lr()[0]:.3e},{loss_G.item():.4f},{loss_D.item():.4f}, , , , \n")

                del loss_D
                del loss_G
                del loss_gan
                del loss_recon
                del loss_real
                del loss_fake
                del fake_img
                del img_data
                del label_data
        
        if (epoch+1)%10==0:
                Test_model_metrics(model_img, val_loader, log_file, device)
                draw_on_test(model_img, val_loader, f"{save_name}/pict_{changes}", epoch, device, name='mmgsn')
                save(model_img, epoch, ckpt_path)