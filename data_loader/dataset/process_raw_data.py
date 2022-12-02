# # # # # # # # # This does the raw image processing from .dng # # # # # # # # 
import os
import numpy as np
from tqdm import tqdm
import rawpy


data_dir = "/media/data4b/haleyso/HDRPlusData/results_20161014"
all_names = os.listdir(data_dir)
for name in tqdm(all_names, ascii=True):
    folder = os.path.join(data_dir, name)
    path = os.path.join(folder, 'merged.dng')
    save_path = os.path.join(folder, 'raw_demosaiced.npy')

    with rawpy.imread(path) as raw:
        '''
            - use the given white balance
            - no auto bright
            - keep in raw colorspace
            - output bits is 16
        '''
        raw_demosaiced = raw.postprocess(gamma=(1,1), use_camera_wb=True, no_auto_bright=True, output_color=rawpy.ColorSpace(0), output_bps=16)
        # print(raw_demosaiced.dtype) # uint16
        np.save(save_path, raw_demosaiced)

