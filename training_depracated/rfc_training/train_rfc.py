import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import functools
import random


training_set_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_9")
output_dir = Path("D:/WaterBodyExtraction/WaterPolyData/rfc")
rfc_name = "rfc_9_e"


def sample(filenames, batch_load_size):
    total_positive_sample = pd.DataFrame()
    total_negative_sample = pd.DataFrame()

    for i in range(0, len(filenames), batch_load_size):
        positive_set = pd.DataFrame()
        negative_set = pd.DataFrame()
        take = batch_load_size if (
            i + batch_load_size < len(filenames)) else len(filenames) - i
        for filename in filenames[i:i+take]:
            training_set = pd.read_csv(
                training_set_dir.joinpath(filename)).iloc[:, 1:]
            positive_set = positive_set.append(
                training_set[training_set.label == False].sample(frac=0.2))
            negative_set = negative_set.append(
                training_set[training_set.label == True])
            print("{} positive samples taken".format(len(positive_set)))
            print("{} loaded...".format(filename))

        print("Sampling...")
        ratio = 1  # int(len(negative_set) / (len(positive_set) * 5))
        for i in range(ratio):
            total_positive_sample = total_positive_sample.append(positive_set)
        total_negative_sample = total_negative_sample.append(negative_set.sample(
            frac=((len(positive_set) * ratio) / len(negative_set))))

        print('\n')
        print("{} positive samples taken".format(len(total_positive_sample)))
        print("{} negative samples taken".format(len(total_negative_sample)))

    return total_positive_sample.append(total_negative_sample)


rfc = RandomForestClassifier(
    n_estimators=5, min_samples_leaf=3)

batch_load_size = 12
filenames = os.listdir(training_set_dir)
random.shuffle(filenames)
training_size = int(len(filenames) * 0.7)
training_filenames = filenames[:training_size]
test_filenames = filenames[training_size:]

print("Loading training sets...")
training_sample = sample(training_filenames, batch_load_size)
print("Loading test sets...")
test_sample = sample(test_filenames, batch_load_size)

X_train = training_sample.drop('label', axis=1)
y_train = list(map(int, training_sample['label']))

X_test = test_sample.drop('label', axis=1)
y_test = list(map(int, test_sample['label']))

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
