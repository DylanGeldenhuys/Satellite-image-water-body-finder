from scipy.ndimage.morphology import binary_closing, binary_opening
from water_body_finder.utilities.post_processing import reduce_noise, smooth
import numpy as np
import matplotlib.pyplot as plt
from water_body_finder.utilities.create_polygon import get_boundary, order_points_simply
import rasterio
from rasterio.plot import reshape_as_image
from pathlib import Path
from water_body_finder.utilities.create_polygon import get_boundary,order_points_simply
import os
import cv2
import sys

clear = lambda: os.system('cls')

tifs = Path('/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs')
output = Path('/media/ds/New Volume/Waterbody_Project/eval_images')
directory = Path('/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines')
label = Path('/media/ds/New Volume/Waterbody_Project/new_labels')

for filename in os.listdir(directory):  
    # save image
    raster = reshape_as_image(rasterio.open(tifs.joinpath(filename.replace('geojson','tif'))).read())
    plt.figure(figsize=(7,7))
    plt.imshow(raster)
    print(raster.shape)



    # save label
    label_data = np.load(label.joinpath(filename.replace('geojson','npy')))
    label_data_alpha = np.zeros(
             (label_data.shape[0], label_data.shape[1], 4), dtype=np.uint8)

    label_data_alpha[label_data == 0] = [255, 17, 0, 22]
    

    label_data_alpha[label_data == 1] = [0, 0, 0, 0]
    #Image.fromarray(label_data_alpha)
    plt.imshow(label_data_alpha)
    plt.title('{}'.format(filename))
    plt.savefig(output.joinpath('{}'.format(filename[:-8] + '_raster.png')))
    plt.close()
    print('saved raster')
    #fig, (ax0) = plt.subplots(ncols=1, sharex=True, sharey=True)

    #ax0.imshow(image_data)
    #ax0.imshow(label_data_alpha)
    

    