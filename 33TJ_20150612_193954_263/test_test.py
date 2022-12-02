import imageio
import rawpy
import cv2


dng_path = '/media/data4b/haleyso/HDRPlusData/results_20161014/33TJ_20150615_050210_628/merged.dng'
gpp_path = '/home/haleyso/HDR_pixel/33TJ_20150612_193954_263/final.jpg'

with rawpy.imread(dng_path) as raw:
    dng_image = raw.raw_image.copy()

gpp_image = cv2.imread(gpp_path)
gpp_image = cv2.cvtColor(gpp_image, cv2.COLOR_BGR2RGB)
print(dng_image.shape, gpp_image.shape)

