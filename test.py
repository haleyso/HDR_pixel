import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import cv2
import os
import numpy as np
import lpips
import kornia
import sys

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

def main(config):
    logger = config.get_logger('test')
    
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            raw_image =  sample['raw_image'].to(device)
            jpg_image =  sample['jpg_image'].to(device)
            name = sample['name'][0]
            save_dir = '/home/haleyso/HDR_pixel/Final_Project'
            save_name = os.path.join(save_dir, name+'LabL2.png')

            # if config["data_loader"]["args"]["data_type"] == 'lab':
            #     raw_image = RGB2Lab(raw_image)
            output = model(raw_image)
            # if config["data_loader"]["args"]["data_type"] == 'lab':
            #     output = Lab2RGB(output)
            print("max:", output.max())

            out_save = output.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255
            jpg_image_save = jpg_image.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255
            out_save = out_save.astype(np.uint8)
            jpg_image_save = jpg_image_save.astype(np.uint8)
            print(out_save.min(), out_save.max(), out_save.shape)
            out_save = cv2.cvtColor(np.hstack([out_save,jpg_image_save]), cv2.COLOR_RGB2BGR)
            print(save_name)
            cv2.imwrite(save_name, out_save)


            batch_size = 1
            for i, metric in enumerate(metric_fns):
                met =  metric(output, jpg_image)
                total_metrics[i] += met * batch_size
                # print(i,met )
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)