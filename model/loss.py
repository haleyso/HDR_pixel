import torch
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as msssim
import lpips
import sys
import kornia


# L1, L2, SSIM_loss, MSSSIM_loss, lpips_loss, lab_L1, lab_L2
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

# RGB metrics
def L1(output, target, device):
    loss =F.l1_loss(output, target)
    return loss

def L2(output, target, device):
    return F.mse_loss(output, target)

def SSIM_loss(output, target, device):
    ssim_inst = ssim(data_range=1.0).to(device)
    d = 1.0 - ssim_inst(output, target)
    return d

def MSSSIM_loss(output, target, device):
    msssim_inst = msssim(data_range=1.0).to(device)
    d = 1.0 - msssim_inst(output, target)
    return d


def lpips_loss(output, target, lpips_loss_fn):
    im0 = output - 1.0
    im1 = target - 1.0
    # scale images to [-1,1] and should be shape Nx3xHxW
    d = lpips_loss_fn.forward(im0,im1)
    return torch.mean(d)


# Lab Metrics
def lab_L1(output,target, device):  
    o_lab = RGB2Lab(output)
    t_lab = RGB2Lab(target)
    loss = F.l1_loss(o_lab,t_lab)
    return loss


def lab_L2(output,target, device):  
    o_lab = RGB2Lab(output)
    t_lab = RGB2Lab(target)
    loss = F.mse_loss(o_lab,t_lab)
    return loss


# def hsv_L1(output,target, device):
#     return F.l1_loss(kornia.color.rgb_to_hsv(output), kornia.color.rgb_to_hsv(target))

# def saturation_L1(output,target, device): # saturation and value loss
#     hsv_output = kornia.color.rgb_to_hsv(output)
#     hsv_target = kornia.color.rgb_to_hsv(target)

#     loss = F.l1_loss(hsv_output[:,1::,:,:], hsv_target[:,1::,:,:])

#     return loss


# def lab_angle(output,target, device):  
#     o_lab = RGB2Lab(output)
#     t_lab = RGB2Lab(target)
#     loss = 0
#     return loss
