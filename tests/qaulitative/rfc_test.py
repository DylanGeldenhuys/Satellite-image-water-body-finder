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


training_set_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/testing_sets/set_8")
rfc_src = Path(
    "D:/WaterBodyExtraction/WaterPolyData/rfc/rfc_8_d.p")

f = open(rfc_src, 'rb')
rfc = pickle.load(f)

filename = "2531DA03_6050_50.csv"  # os.listdir(training_set_dir)[14]
training_set = pd.read_csv(
    training_set_dir.joinpath(filename)).iloc[:, 1:]

features = training_set.drop('label', axis=1)
labels = list(map(int, training_set['label']))

predictions = rfc.predict(features)

print('\n')
print("=== Classification Report ===")
print(classification_report(labels, predictions))
print('\n')

prediction_img = predictions.reshape(1874, 2000)
label_img = np.array(labels).reshape(1874, 2000)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True)

ax0.imshow(prediction_img)
ax0.set_title("Prediction")
ax0.axis("off")

ax1.imshow(label_img)
ax1.set_title("Label")
ax1.axis("off")

fig.tight_layout()
plt.show()
