import numpy as np
import matplotlib.pyplot as plt
import os
from utils.dataset_png import * 
from utils.metrics_calculate_2D import cal_metric_list
from torch.utils.data import Dataset, DataLoader
import time
def get_time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

@torch.no_grad()    
def draw_on_test(model_img, test_dataloader, picture_save, epoch, device, num=8, name=''):
    model_img.eval()
    data_batch  = next(iter(test_dataloader))

    input_img, target_img = data_batch[:, :2, :, :].to(device), data_batch[:, 2:3, :, :].to(device)
    batch_size, img_W = target_img.shape[0], target_img.shape[2]
    if num>batch_size:
        num = batch_size-1
    import random
    random_index = random.sample([i for i in range(batch_size)], num)
    input_img, target_img = input_img[random_index], target_img[random_index]

    fake_img = model_img(input_img)

    fake_img, target_img, input_img = fake_img.reshape(-1, img_W).detach().cpu(), target_img, input_img

    input_img = input_img.transpose(1,2).reshape(-1, 2*img_W).detach().cpu()
    target_img = target_img.reshape(-1, img_W).detach().cpu()
    target_fake_img = torch.concat([input_img, target_img, fake_img], dim=1)
    plt.figure(figsize=(10,20), dpi=150)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.imshow(target_fake_img, cmap= 'gray')

    os.makedirs(picture_save, exist_ok=True) 
    plt.savefig(f"{picture_save}{name}_epoch_{epoch}.png")

    plt.close()




@torch.no_grad()
def Test_model_metrics(model_img, test_dataloader:DataLoader, log_file, device, test_all = False): #

    AUC_imgs, MSE_imgs, SSIM_imgs, PSNR_imgs = [], [], [], []
    model_img.eval()

    for data_batch in test_dataloader:
        input_img, target_img = data_batch[:, :2, :, :].to(device), data_batch[:, 2:3, :, :].to(device)
 
        fake_img = model_img(input_img)

        fake_img, target_img = fake_img.squeeze(dim=1).detach().cpu().numpy(), target_img.squeeze(dim=1).detach().cpu().numpy()
        AUC_list, MSE_list, SSIM_list,  PSNR_list = cal_metric_list(fake_img, target_img, method='0-1', norm=True)
        AUC_imgs.extend(AUC_list)
        MSE_imgs.extend(MSE_list)
        SSIM_imgs.extend(SSIM_list)
        PSNR_imgs.extend(PSNR_list)
        if len(AUC_imgs) >= 64 and not test_all:
            break

    AUC_imgs, MSE_imgs, SSIM_imgs, PSNR_imgs = np.mean(AUC_imgs), np.mean(MSE_imgs), np.mean(SSIM_imgs), np.mean(PSNR_imgs)
    log_file.write(f"AUC_imgs: {AUC_imgs}, MSE_imgs: {MSE_imgs}, SSIM_imgs: {SSIM_imgs}, PSNR_imgs: {PSNR_imgs} \n")

