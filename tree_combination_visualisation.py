import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import functools
import pickle
from water_body_finder import extract_features, calc_img_dimensions
import pandas as pd
from sklearn.metrics import classification_report
import rasterio
from rasterio.plot import reshape_as_image

filename = "2531DA03"

rfc_dir = Path("D:/WaterBodyExtraction/WaterPolyData/rfc")
rfc_version = "rfc_2"
num_of_trees = 10

training_set_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_2")
label_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/label_data")
image_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/image_data")


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


# load image
raster_image_data = rasterio.open(
    image_data_directory.joinpath(filename + '.tif')).read()
image_data = reshape_as_image(raster_image_data)
print("Image loaded...")

# load features
training_set = pd.read_csv(
    training_set_dir.joinpath(filename + '.csv')).iloc[:, 1:]
features = training_set.drop('label', axis=1)
labels = list(map(int, training_set['label']))
print("Features loaded...")

# load forests
rfc_ls = []
for i in range(num_of_trees):
    f = open(rfc_dir.joinpath(
        rfc_version + "_{}".format(i) + '.p'), 'rb')
    rfc = pickle.load(f)
    rfc_ls.append(rfc)
print("Forests loaded...")

fig = plt.figure()
fig.add_subplot((num_of_trees + 1), 1, 1)
plt.imshow(np.load(Path(label_dir.joinpath(filename + '.npy'))))

width, height = calc_img_dimensions(image_data, 5, 15)

for i in range(num_of_trees):
    rfc_combined = functools.reduce(combine_rfs, rfc_ls[0:(i + 1)])
    prediction = rfc_combined.predict(features)

    print("=== Classification Report ===")
    print(classification_report(labels, prediction))
    print('\n')
    result = np.reshape(prediction, [height - 199, width - 199])

    fig.add_subplot((num_of_trees + 1), 1, (i + 2))
    plt.imshow(result)

plt.show()
