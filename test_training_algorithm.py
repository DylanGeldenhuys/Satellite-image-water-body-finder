from water_body_finder import find_waterbody_mask
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image
import pickle
import matplotlib.pyplot as plt
import numpy as np

image_data_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/image_data")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_1")
rfc_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/rfc/rfc_1.p")

filename = "2531DA03"

raster_image_data = rasterio.open(
    image_data_dir.joinpath(filename + '.tif')).read()
image_data = reshape_as_image(raster_image_data)

f = open(rfc_dir, 'rb')
rfc = pickle.load(f)

result = find_waterbody_mask(image_data, rfc, 5, 15)
np.save(output_dir.joinpath(output_dir.joinpath(filename + '.npy')), result)
