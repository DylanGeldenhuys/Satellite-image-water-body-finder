import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import functools

training_set_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_2")
output_dir = Path("D:/WaterBodyExtraction/WaterPolyData/rfc")
rfc_name = "rfc_3"

print("Loading training set...")
training_set = pd.read_csv(training_set_dir.joinpath(
    os.listdir(training_set_dir)[29])).iloc[:, 1:]
print("Creating training split...")
X = training_set.drop('label', axis=1)
y = list(map(int, training_set['label']))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=66)

print("Training forest...")
rfc = RandomForestClassifier(n_estimators=5, min_samples_leaf=3)
rfc.fit(X_train, y_train)

print("Pickling forest...")
pickle.dump(rfc, open(output_dir.joinpath(rfc_name + '.p'), "wb"))

print("Predicting...")
rfc_predict = rfc.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
