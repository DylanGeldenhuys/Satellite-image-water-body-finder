import numpy as np
from pathlib import Path
from water_body_finder.utilities import reduce_noise, get_boundary, order_edge_points, smooth
from water_body_finder.train import create_label
from tests import MAEi, MAEj
import matplotlib.pyplot as plt
import pandas as pd

prediction_dir = Path(
    "/media/ds/New Volume/Waterbody_Project/OUTPUT/polygons/version_0")
image_dir = Path('/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs')

geo_data_label_dir = Path('/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines')

output_dir = Path(
    "/media/ds/New Volume/Waterbody_Project/OUTPUT")

filenames = os.listdir(prediction_dir)
empty_list = []
for filename in filenames:
    raster_object = rasterio.open(image_dir.joinpath(filename.replace('npy','tif')))
    with open(geo_data_label_dir.joinpath(filename.replace('npy', 'geojson'))) as f:
        geo_data = json.load(f) 
    label_mask = create_label(rasterio_object, geo_data)
    label_boundary = get_boundary(label_mask)
    prediction_boundary = np.load(prediction_dir.joinpath(filename))
    empty_list.append([filename[:-4],MAEi(prediction_boundary, label_boundary), MAEj(label_boundary, prediction_boundary) ])

maei = pd.DataFrame(empty_list)
print(maei)

