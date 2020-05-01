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
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_8")
output_dir = Path("D:/WaterBodyExtraction/WaterPolyData/rfc")
rfc_name = "rfc_8"

rfc = RandomForestClassifier(
    n_estimators=5, min_samples_leaf=3)

full_training_set = pd.DataFrame()
print("Loading training sets...")
for filename in os.listdir(training_set_dir):
    training_set = pd.read_csv(
        training_set_dir.joinpath(filename)).iloc[:, 1:]
    full_training_set = full_training_set.append(training_set)
    print("{} loaded...".format(filename))

print("Sampling...")
positive_sample = full_training_set[full_training_set.label == False]
negative_set = full_training_set[full_training_set.label == True]
negative_sample = negative_set.sample(
    len(positive_sample) / len(negative_set))

print("{} positive samples taken".format(len(positive_sample)))
print("{} negative samples taken".format(len(negative_sample)))

full_sample = positive_sample.append(negative_sample)

X = full_sample.drop('label', axis=1)
y = list(map(int, full_sample['label']))

print("Splitting training set...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=66)

print('\n')
print("Training forest...")
rfc.fit(X_train, y_train)

print('\n')
print("Pickling forest...")
pickle.dump(rfc, open(output_dir.joinpath("{}.p".format(rfc_name)), "wb"))

print('\n')
print("Predicting...")
rfc_predict = rfc.predict(X_test)

print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')

text_file = open(output_dir.joinpath(
    "{0}_classifiction_report.txt".format(rfc_name)), "w")
text_file.write("=== Classification Report ===")
text_file.write("\n")
text_file.write(str(classification_report(y_test, rfc_predict)))
text_file.close()
