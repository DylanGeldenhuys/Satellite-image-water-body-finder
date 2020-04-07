from water_body_finder import find_waterbody_mask, extract_features_v3
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image
import pickle
import matplotlib.pyplot as plt
import numpy as np

image_data_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/image_data")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_3")
rfc_src = Path(
    "D:/WaterBodyExtraction/WaterPolyData/rfc/rfc_3_1.p")

filename = "2531DA03"

raster_image_data = rasterio.open(
    image_data_dir.joinpath(filename + '.tif')).read()
image_data = reshape_as_image(raster_image_data)

f = open(rfc_src, 'rb')
rfc = pickle.load(f)

result = find_waterbody_mask(image_data, rfc, extract_features_v3)
np.save(output_dir.joinpath(output_dir.joinpath(filename + '_1.npy')), result)
