import numpy as np
from pathlib import Path
from water_body_finder.utilities import reduce_noise, get_boundary, order_edge_points, smooth
from tests import MAEi, MAEj
import matplotlib.pyplot as plt

filename = "2531DA03"
prediction_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_3")
#label_dir = Path(
#    "D:/WaterBodyExtraction/WaterPolyData/label_data")
geo_data_label = Path('')
with open(geo_data_src) as f:
    geo_data = json.load(f)

output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/evaluations")
data_resolution = 3
data_padding = 9

load_boundaries = True

if (load_boundaries):
    prediction_boundary = np.load(output_dir.joinpath(
        "{}_prediction_boundary_b.npy".format(filename)))
    label_boundary = np.load(output_dir.joinpath(
        "{}_label_boundary_b.npy".format(filename)))

if (load_boundaries == False):
    print("Loading prediction...")
    # load prediction
    prediction = np.load(prediction_dir.joinpath(filename + '.npy'))
    print("Reducing noise...")
    # reduce noise
    prediction = reduce_noise(prediction, 500, 200)
    prediction = smooth(prediction)

    print("Loading label...")
    # load label mask
    label = np.load(label_dir.joinpath(filename + '.npy'))

    # transform label and prediction to match dimensions
    transformed_prediction = np.repeat(
        np.repeat(prediction, data_resolution, axis=0), data_resolution, axis=1)

    padding = int(9/2)
    transformed_label = label[padding:-padding, padding:-padding]

    # get boundaries
    print("Calculating prediction boundary...")
    prediction_boundary = get_boundary(
        transformed_prediction)
    print("Calculating label boundary...")
    label_boundary = get_boundary(transformed_label)

    print("Saving...")
    np.save(output_dir.joinpath(
        "{}_prediction_boundary_b.npy".format(filename)), prediction_boundary)
    np.save(output_dir.joinpath(
        "{}_label_boundary_b.npy".format(filename)), label_boundary)

maei = MAEi(prediction_boundary, label_boundary)
maeij = MAEj(label_boundary, prediction_boundary)

print("Evaluating...")
print("MAEi: {0}".format(maei))
print("MAEj:{0}".format(maeij))

plt.gca().invert_yaxis()
plt.scatter(prediction_boundary[:, 0], prediction_boundary[:, 1])
plt.scatter(label_boundary[:, 0], label_boundary[:, 1])
plt.show()
