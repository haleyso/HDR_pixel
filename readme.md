# HDR_pixel

The dataset can be downloaded at [https://hdrplusdata.org/dataset.html](https://hdrplusdata.org/dataset.html). I used the curated dataset of 153 images.

--dataset<br>
----image_name<br>
------merged.dng<br>
------final.jpg<br>

To first prep the dataset, run the process_raw_data.py file (found in path /data_loader/dataset/process_raw_data.py). This will do some raw processing to demosaic the images and make a raw_demosaiced.npy for each image. 


Create config file (see config/config.json for example)

To train:

```python train.py -c [config_file] ```


Once you have the trained model, to run on the test set,

'''```python train.py -c [test_config_file] -r [path/to/checkpoint-epoch.pth]``` 


