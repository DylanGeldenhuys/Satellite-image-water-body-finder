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
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_8")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_4")
rfc_src = Path(
    "D:/WaterBodyExtraction/WaterPolyData/rfc/rfc_8.p")

f = open(rfc_src, 'rb')
rfc = pickle.load(f)

filename = "2828BB06_6050_6050.csv"  # os.listdir(training_set_dir)[14]
training_set = pd.read_csv(
    training_set_dir.joinpath(filename)).iloc[:, 1:]

features = training_set.drop('label', axis=1)
labels = list(map(int, training_set['label']))

predictions = rfc.predict(features)

print('\n')
print("=== Classification Report ===")
print(classification_report(labels, predictions))
print('\n')

prediction_img = predictions.reshape(2000, 2000)
label_img = np.array(labels).reshape(2000, 2000)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True)

ax0.imshow(label_img)
ax0.set_title("Label")
ax0.axis("off")

ax1.imshow(prediction_img)
ax1.set_title("Prediction")
ax1.axis("off")

fig.tight_layout()
plt.show()
