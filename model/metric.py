import torch
# import skimage
from skimage.metrics import mean_squared_error as skimage_mse
from skimage.metrics import structural_similarity as skimage_ssim
import sys
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mse(output, target):
    with torch.no_grad():
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        mse_score = skimage_mse(target, output)
        
    return mse_score

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
        # print(ssim_score)
        # sys.exit()
    return ssim_score



