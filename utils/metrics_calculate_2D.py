import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.filters import  threshold_multiotsu
from sklearn.metrics import roc_auc_score

def norm_layerHW(imgdata, method = '8bit'): 
    layers = imgdata.shape[0]
    max_value = np.max(imgdata.reshape(layers,-1),axis=-1).reshape(layers,1,1)
    min_value = np.min(imgdata.reshape(layers,-1),axis=-1).reshape(layers,1,1)
    imgdata = (imgdata-min_value)/(max_value-min_value+1e-6) 
    if method == '8bit':
        imgdata = (255 * imgdata).astype(np.uint8) 
    elif method == '0-1':
        pass
    elif method == '+-1':
        imgdata = 2*imgdata -1
    return imgdata


def tanhnorm_layerHW(imgdata, method = '8bit'): 

    imgdata = np.tanh(imgdata) 

    if method == '8bit':
        imgdata = (255 * (imgdata+1)/2).astype(np.uint8) 
    if method == '0-1':
        imgdata = (imgdata+1)/2
    if method == '+-1':
        pass
    return imgdata

def norm_HW(imgdata, method = '8bit'): 
    max_value, min_value = imgdata.max(), imgdata.min()
    imgdata = (imgdata-min_value)/(max_value-min_value+1e-10)
    if method == '8bit':
        imgdata = (255 * imgdata).astype(np.uint8) 
    if method == '0-1':
        pass
    if method == '+-1':
        imgdata = 2*imgdata -1
    return imgdata

def auc_img(fake_img, true_img, classes = 4, method = '8bit'):

    try:
        thresholds = threshold_multiotsu(true_img, classes=classes) 
        binary = true_img >= thresholds[-1]
        y_true, y_pred = binary.flatten(), fake_img.flatten()
        if method == '8bit':
            y_pred = y_pred/255
        if method == '0-1':
            pass
        if method == '+-1':
            y_pred = (y_pred +1)/2
        
        return roc_auc_score(y_true, y_pred)
    except:
        return np.nan


def ssim_img(true_img, fake_img, method = '8bit'):
    if method == '8bit':
        score = ssim(true_img, fake_img)
    if method == '0-1':
        score = ssim(true_img, fake_img, data_range=1.0)
    if method == '+-1':
        score = ssim(true_img, fake_img, data_range=2.0)
    return score

def psnr_img(true_img, fake_img, method = '8bit'):
    if method == '8bit':
        score = psnr(true_img, fake_img)
    if method == '0-1':
        score = psnr(true_img, fake_img, data_range=1.0)
    if method == '+-1':
        score = psnr(true_img, fake_img, data_range=2.0)
    return score


def cal_metric_list(fake_img, target_img, method = '+-1', norm = True): #256@256
    if norm:
        fake_img, target_img = norm_layerHW(fake_img, method=method), norm_layerHW(target_img, method=method)
    else:
        method = '+-1'
    layers = target_img.shape[0]
    AUC = [auc_img(target_img[i], fake_img[i], method=method) for i in range(layers)]
    MSE = [mse(target_img[i], fake_img[i]) for i in range(layers)]
    SSIM = [ssim_img(target_img[i], fake_img[i], method=method) for i in range(layers)] 
    PSNR = [psnr_img(target_img[i], fake_img[i], method=method) for i in range(layers)] 

    return AUC, MSE, SSIM, PSNR