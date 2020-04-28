import rasterio
from rasterio import mask
from shapely import geometry
import json
import numpy as np
from pathlib import Path
import os

geo_data_directory = Path("D:/WaterBodyExtraction/WaterPolyData/geo_data")
image_data_directory = Path("D:/WaterBodyExtraction/WaterPolyData/image_data")

output_directory = Path("D:/WaterBodyExtraction/WaterPolyData/label_data")

for filename in os.listdir(geo_data_directory)[85:]:
    # load files
    with open(geo_data_directory.joinpath(filename)) as f:
        geo_data = json.load(f)

    image_data = rasterio.open(
        image_data_directory.joinpath(filename.replace("geojson", "tif")))

    # extract shapes
    shapes = []
    for feature in geo_data['features']:
        shapes.append(geometry.Polygon(
            [[p[0], p[1]] for p in feature['geometry']['coordinates']]))

    # create mask from shapes
    mask = rasterio.mask.raster_geometry_mask(image_data, shapes)[0]

    # save mask
    np.save(output_directory.joinpath(
        filename.replace("geojson", "npy")), mask)
