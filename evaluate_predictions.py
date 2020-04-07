import numpy as np
from pathlib import Path
from water_body_finder.utilities import reduce_noise, get_boundary, order_edge_points
from tests import MAEi, MAEj

filename = ""
prediction_dir = Path("")
label_dir = Path("")
output_dir = Path("")
data_resolution = 3
data_padding = 9


# load prediction
prediction = np.load(prediction_dir.joinpath(filename + '.npy'))
# reduce noise
prediction = reduce_noise(prediction, 500, 200)

# load label mask
label = np.load(label_dir.joinpath(filename + '.npy'))

# transform label and prediction to match dimensions
transformed_prediction = np.repeat(
    np.repeat(prediction, data_resolution, axis=0), data_resolution, axis=1)

padding = int(9/2)
transformed_label = label[padding:-padding, padding:-padding]

# get boundaries
prediction_boundary = get_boundary(transformed_prediction)
label_boundary = get_boundary(transformed_label)

np.save(output_dir.joinpath(
    "{}_prediction_boundary.npy".format(filename)), prediction_boundary)
np.save(output_dir.joinpath(
    "{}_label_boundary.npy".format(filename)), label_boundary)

print("MAEi: {}".format(MAEi(prediction_boundary, label_boundary)))
print("MAEj:{}".format(MAEj(label_boundary, prediction_boundary)))
