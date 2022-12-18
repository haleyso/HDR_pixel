import torch
import skimage
from skimage.metrics import mean_squared_error as skimage_mse
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import sys
import numpy as np
import kornia
from torchmetrics import PeakSignalNoiseRatio as tm_psnr

# metrics: mse, psnr, ssim, deltaE

#----------------------------------------------------------------------
def normalize_lab(image):
    '''
        - divides the Luminance channel by 100
        - divides the other 2 channels by 127
    '''
    # image: 4,3,100,100
    n_image = image.clone()
    n_image[:,0,:,:] = image[:,0,:,:]/100.
    n_image[:,1,:,:] = image[:,1,:,:]/127.
    n_image[:,2,:,:] = image[:,2,:,:]/127.

    return n_image

def denormalize_lab(image):
    '''
        - mulitples the Luminance channel by 100
        - multiplies the other 2 channels by 127
    '''
    # image: 4,3,100,100
    n_image = image.clone()
    n_image[:,0,:,:] = image[:,0,:,:]*100.
    n_image[:,1,:,:] = image[:,1,:,:]*127.
    n_image[:,2,:,:] = image[:,2,:,:]*127.

    return n_image

def RGB2Lab(image):
    out = kornia.color.rgb_to_lab(image)
    out = normalize_lab(out)
    return out

def Lab2RGB(image):
    out = denormalize_lab(image)
    out = kornia.color.lab_to_rgb(image)
    return out

#_-----------------------------------------------------------------------

def mse(output, target):
    with torch.no_grad():
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        mse_score = skimage_mse(target, output)
        
    return mse_score


# def hsv_mse(output, target):
#     with torch.no_grad():
#         target = kornia.color.rgb_to_hsv(target)
#         output = kornia.color.rgb_to_hsv(output)
#         target = target.cpu().detach().numpy()
#         output = output.cpu().detach().numpy()
#         hsv_mse_score = skimage_mse(target, output)
#     return hsv_mse_score
    

def psnr(output, target):
    with torch.no_grad():
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        psnr_score = skimage_psnr(target, output, data_range=int(target.max()))
    return psnr_score


def ssim(output, target):
    with torch.no_grad():
        ssim_score = 0
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()

        target = np.transpose(target, (0,2,3,1))
        output = np.transpose(output, (0,2,3,1))
        for i in range(target.shape[0]):
            ssim_score += skimage_ssim(target[i,:,:,:], output[i,:,:,:], multichannel=True)
        ssim_score = ssim_score/target.shape[0]
    return ssim_score


# aka L2 in lab space
def deltaE(output, target):
    with torch.no_grad():
        target = RGB2Lab(target)
        output = RGB2Lab(output)
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        score = skimage_mse(target, output)
    return score

def lpips_loss(output, target, lpips_loss_fn):
    with torch.no_grad():
        im0 = output - 1.0
        im1 = target - 1.0
        # scale images to [-1,1] and should be shape Nx3xHxW
        d = lpips_loss_fn.forward(im0,im1)
    return torch.mean(d)
