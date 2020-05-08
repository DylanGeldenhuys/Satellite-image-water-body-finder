import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

import pickle
from pathlib import Path
import functools
import random


training_set_dir = Path(
    "/media/ds/New Volume/Waterbody_Project/Training_set_1")
output_dir = Path("/media/ds/New Volume/Waterbody_Project/RFC'S")
rfc_name = "rfc_Hyperparam_tuning_text"

print("Loading training sets...")
total_positive_sample = pd.DataFrame()
total_negative_sample = pd.DataFrame()

batch_load_size = 12
filenames = os.listdir(training_set_dir)
random.shuffle(filenames)

for i in range(0, len(filenames), batch_load_size):
    positive_set = pd.DataFrame()
    negative_set = pd.DataFrame()
    for filename in filenames[i:i+batch_load_size]:
        training_set = pd.read_csv(
            training_set_dir.joinpath(filename)).iloc[:, 1:]
        positive_set = positive_set.append(
            training_set[training_set.label == False])
        negative_set = negative_set.append(
            training_set[training_set.label == True])
        print("{} positive samples taken".format(len(positive_set)))
        print("{} loaded...".format(filename))

    print("Sampling...")
    ratio = int(len(negative_set) / (len(positive_set) * 25))
    for i in range(ratio):
        total_positive_sample = total_positive_sample.append(positive_set)
    total_negative_sample = total_negative_sample.append(negative_set.sample(
        frac=((len(positive_set) * ratio) / len(negative_set))))

    print('\n')
    print("{} positive samples taken".format(len(total_positive_sample)))
    print("{} negative samples taken".format(len(total_negative_sample)))

full_sample = total_positive_sample.append(total_negative_sample)
full_sample = full_sample.sample(frac=1)

X = full_sample.drop('label', axis=1)
y = list(map(int, full_sample['label']))
del full_sample,total_negative_sample,total_positive_sample,positive_set,negative_set

print("Splitting training set...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=66)

print('\n')
print("Training forest...")
#rfc.fit(X_train, y_train)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rfc = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
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

print("Cross validating...")
rfc_cv_score = cross_val_score(
    rfc, X_train, y_train, cv=5, scoring='roc_auc')

text_file = open(output_dir.joinpath(
    "{0}_classifiction_report.txt".format(rfc_name)), "w")
text_file.write("=== Classification Report ===")
text_file.write('\n')
text_file.write(str(classification_report(y_test, rfc_predict)))
text_file.write('\n')
text_file.write('\n')
text_file.write("=== Confusion Matrix ===")
text_file.write('\n')
text_file.write(str(confusion_matrix(y_test, rfc_predict)))
text_file.write('\n')
text_file.write('\n')
text_file.write("=== All AUC Scores ===")
text_file.write('\n')
text_file.write(str(rfc_cv_score))
text_file.write('\n')
text_file.write('\n')
text_file.write("=== Mean AUC Score ===")
text_file.write('\n')
text_file.write(
    "Mean AUC Score - Random Forest: {}".format(rfc_cv_score.mean()))
text_file.write('\n')
text_file.write('\n')
text_file.write("=== Feature Importances ===")
text_file.write(
    "{}".format(rfc.feature_importances_))
text_file.close()
