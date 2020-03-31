# Here we create the T files for training and testing


# Inputs: Folder


import os
import rasterio
import cv2
from feature_data_functions import create_training_samples
import pandas as pd
from rasterio.plot import reshape_as_image

image_dir = "/media/ds/New Volume/Waterbody_Project/Tiffs"
mask_dir = "/home/ds/Projects/satalite-image-water-body-finder/masks"

tiff_ls = os.listdir(image_dir)

index = 0
tiff_length = len(tiff_ls)

for filename in tiff_ls[0:25]:
    raster = rasterio.open(image_dir + "/" + filename).read()
    img = reshape_as_image(raster)

    mask = cv2.imread(mask_dir + "/" + filename.replace("tif", "jpg"))

    training_samples = create_training_samples(5, img, mask, filename[0:-4])

    training_df = pd.DataFrame(training_samples)
    training_df.columns = ['image_name', 'variance', 'color1', 'color2', 'color3','variance2','color21','color22','color23', 'label']

    index += 1
    print("{}%".format((index / tiff_length )* 100))

    training_df.to_csv('/home/ds/Projects/satalite-image-water-body-finder/T2_data/{}.csv'.format(filename[0:-4]))



