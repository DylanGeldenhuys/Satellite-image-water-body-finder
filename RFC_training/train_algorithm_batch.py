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
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_4")
output_dir = Path("D:/WaterBodyExtraction/WaterPolyData/rfc")
rfc_name = "rfc_4"

batch_size = 10
rfc_index = 2
num_iterations = 1

#rfc = None
f = open(output_dir.joinpath("rfc_4_1.p"), 'rb')
rfc = pickle.load(f)

if rfc == None:
    print("No rfc found, creating new rfc...")
    rfc = RandomForestClassifier(
        n_estimators=5, min_samples_leaf=3, warm_start=True)

for i in range(num_iterations):
    full_training_set = pd.DataFrame()
    print("Loading training sets...")
    for filename in (os.listdir(training_set_dir)[(rfc_index * batch_size):(rfc_index * batch_size + batch_size)]):
        training_set = pd.read_csv(
            training_set_dir.joinpath(filename)).iloc[:, 1:]
        full_training_set = full_training_set.append(training_set)
        print("{} loaded...".format(filename))

    X = full_training_set.drop('label', axis=1)
    y = list(map(int, full_training_set['label']))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=66)

    print('\n')
    print("Training forest...")
    rfc.fit(X_train, y_train)
    rfc.n_estimators += 1

    print('\n')
    print("Pickling forest...")
    pickle.dump(rfc, open(output_dir.joinpath(
        rfc_name + "_{}.p".format(rfc_index)), "wb"))

    print('\n')
    print("Predicting...")
    rfc_predict = rfc.predict(X_test)

    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')

    text_file = open(output_dir.joinpath(
        "{0}_{1}_classifiction_report.txt".format(rfc_name, rfc_index)), "w")
    text_file.write("=== Classification Report ===")
    text_file.write("\n")
    text_file.write(str(classification_report(y_test, rfc_predict)))
    text_file.close()
    rfc_index += 1
