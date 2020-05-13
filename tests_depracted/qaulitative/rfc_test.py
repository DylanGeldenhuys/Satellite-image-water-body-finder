import sys
sys.path.append('C:/personal/satalite-image-water-body-finder')  # noqa

from water_body_finder.utilities.post_processing import reduce_noise, smooth

import os
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage.morphology import binary_closing, binary_opening


training_set_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/testing_sets/set_9")
rfc_src = Path(
    "D:/WaterBodyExtraction/WaterPolyData/rfc/rfc_9_b.p")

f = open(rfc_src, 'rb')
rfc = pickle.load(f)

filename = "3319DC01_9050_3050"  # os.listdir(training_set_dir)[14]
shape = [877, 1000]
training_set = pd.read_csv(
    training_set_dir.joinpath(filename + '.csv')).iloc[:, 1:]

features = training_set.drop('label', axis=1)
labels = list(map(int, training_set['label']))

predictions = rfc.predict(features)

print('\n')
print("=== Classification Report ===")
print(classification_report(labels, predictions))
print('\n')

prediction_img = predictions.reshape(shape[0], shape[1])
label_img = np.array(labels).reshape(shape[0], shape[1])
print("Creating image...")
image = list(zip(training_set['color_r'],
                 training_set['color_g'], training_set['color_b']))
image_array = np.array(image).reshape(shape[0], shape[1], 3)
print("Image created")

prediction_close = binary_closing(prediction_img, iterations=2)
prediction_open = binary_opening(prediction_close)
prediction_noise_reduction = reduce_noise(prediction_open, 300, 6000)
prediction_final = smooth(prediction_noise_reduction)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
    ncols=2, nrows=2, sharex=True, sharey=True)

ax0.imshow(image_array)
ax0.set_title("Image")
ax0.axis("off")

ax1.imshow(prediction_img)
ax1.set_title("prediction_img")
ax1.axis("off")

ax2.imshow(prediction_noise_reduction)
ax2.set_title("prediction_noise_reduction")
ax2.axis("off")

ax3.imshow(prediction_final)
ax3.set_title("prediction_noise_reduction")
ax3.axis("off")

fig.tight_layout()
plt.show()
