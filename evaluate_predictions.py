import numpy as np
from pathlib import Path
from water_body_finder.utilities import reduce_noise, get_boundary, order_edge_points

filename = ""
prediction_dir = Path("")
label_dir = Path("")
output_dir = ""
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

