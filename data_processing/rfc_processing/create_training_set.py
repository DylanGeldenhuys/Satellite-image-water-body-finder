import sys
sys.path.append('C:/personal/satalite-image-water-body-finder')  # noqa

import os
from water_body_finder.feature_extraction import entropy
import pandas as pd
from pathlib import Path
import numpy as np
from rasterio.plot import reshape_as_image
import rasterio


# define parameters
image_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/image_data")
label_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/label_data")

output_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_4")

image_label_fix_src = Path(
    "")

# load image label fix
image_label_fix_data = pd.read_csv(image_label_fix_src)

# loop through images
for filename in os.listdir(image_data_directory):
    # load files
    raster_image_data = rasterio.open(
        image_data_directory.joinpath(filename)).read()
    image_data = reshape_as_image(raster_image_data)

    label_data = np.load(label_data_directory.joinpath(
        filename.replace("tif", "npy")))
