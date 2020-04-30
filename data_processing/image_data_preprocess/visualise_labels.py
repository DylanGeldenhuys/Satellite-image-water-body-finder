import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.plot import reshape_as_image
import rasterio
from PIL import Image
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os


image_data_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs")
label_data_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/comp_label")

filename = '2931AC10.tif' #os.listdir(image_data_directory)[4]

# load files
raster_image_data = rasterio.open(
    image_data_directory.joinpath(filename)).read()
image_data = reshape_as_image(raster_image_data)

label_data = np.load(label_data_directory.joinpath(
    filename.replace("tif", "npy")))

label_data_alpha = np.zeros(
    (label_data.shape[0], label_data.shape[1], 4), dtype=np.uint8)

label_data_alpha[label_data == 0] = [255, 17, 0, 50]

label_data_alpha[label_data == 1] = [0, 0, 0, 0]
Image.fromarray(label_data_alpha)

fig, (ax0) = plt.subplots(ncols=1, sharex=True, sharey=True)

ax0.imshow(image_data)
ax0.imshow(label_data_alpha)

#ax1.imshow(image_data)

plt.show()
