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
rfc_name = "rfc_2"
training_set_size = 20


def generate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=5, min_samples_leaf=3)
    rf.fit(X_train, y_train)

    rfc_predict = rf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    return rf


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


rfc_ls = []

for i in range(10):
    print("Complete: {}%".format(i / training_set_size * 100))
    training_set = pd.read_csv(training_set_dir.joinpath(
        os.listdir(training_set_dir)[i])).iloc[:, 1:]
    X = training_set.drop('label', axis=1)
    y = list(map(int, training_set['label']))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=66)

    rfc = generate_rf(X_train, y_train, X_test, y_test)
    pickle.dump(rfc, open(output_dir.joinpath(
        rfc_name + '_' + str(i) + '.p'), "wb"))
    rfc_ls.append(rfc)

rf_clf_combined = functools.reduce(combine_rfs, rfc_ls)

pickle.dump(rf_clf_combined, open(output_dir.joinpath(rfc_name + '.p'), "wb"))
