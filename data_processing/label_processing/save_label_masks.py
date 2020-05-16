import rasterio
from rasterio import mask
from shapely import geometry
import json
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

geo_data_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines")
image_data_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs")

output_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/new_labels")
visualisation_output_directory = Path(
    "/media/ds/New Volume/Waterbody_Project/new_labels_vis")

for filename in os.listdir(geo_data_directory):
    print(filename)
    # load files

    with open(geo_data_directory.joinpath(filename)) as f:
        geo_data = json.load(f)

    image_data = rasterio.open(
        image_data_directory.joinpath(filename.replace("geojson", "tif")))
    

    # extract shapes
    shapes = []

    for feature in geo_data['features']:
        
        for i in range(len(feature['geometry']['coordinates'])):
            #print(feature['geometry']['coordinates'][i])
            shapes.append(geometry.Polygon(
                [[p[0], p[1]] for p in feature['geometry']['coordinates'][i]]))
    # create mask from shapes
    mask = rasterio.mask.raster_geometry_mask(image_data, shapes)[0]

    # save mask
    np.save(output_directory.joinpath(
        filename.replace("geojson", "npy")), mask)
    # 
    Image.fromarray(mask.astype(np.uint8)).save(visualisation_output_directory.joinpath(
                    "{}.png".format(filename[:-8])))
