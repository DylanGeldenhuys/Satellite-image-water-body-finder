import rasterio
from rasterio import mask
from shapely import geometry
import json
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import rasterio
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from .feature_extraction import extract_features
from .utilities import load_window


def train_model(output_dir, training_dir, version="0", sub_version=None):
    rfc_path = Path(output_dir).joinpath('rfc')
    rfc_path.mkdir(parents=True, exist_ok=True)

    training_path = Path(training_dir).joinpath(
        "version_{}".format(version))
    filenames = os.listdir(training_path)

    full_training_set = pd.DataFrame()
    for filename in filenames:
        full_training_set = full_training_set.append(pd.read_csv(
            training_path.joinpath(filename)).iloc[:, 1:])

    positive_samples = full_training_set[full_training_set.label == False]
    negative_samples = full_training_set[full_training_set.label == True]

    final_training_set = pd.DataFrame()
    if (len(positive_samples) > len(negative_samples)):
        positive_samples = positive_samples.sample(
            frac=len(negative_samples) / len(positive_samples))
    if (len(negative_samples) > len(positive_samples)):
        negative_samples = negative_samples.sample(
            frac=len(positive_samples) / len(negative_samples))

    print("Positive: {0} Negative: {1}".format(
        len(positive_samples), len(negative_samples)))
    final_training_set = final_training_set.append(
        positive_samples).append(negative_samples)

    X = final_training_set.drop('label', axis=1)
    y = list(map(int, final_training_set['label']))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    rfc = RandomForestClassifier(
        n_estimators=5, min_samples_leaf=3)

    print("Training forest...")
    rfc.fit(X_train, y_train)

    print('\n')
    print("Pickling forest...")
    pickle.dump(rfc, open(rfc_path.joinpath("rfc_{0}{1}.p".format(
        version, "_{}".format(sub_version) if sub_version != None else "")), "wb"))

    print('\n')
    print("Predicting...")
    rfc_predict = rfc.predict(X_test)

    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')

    print('\n')
    print("Predicting...")
    rfc_training_predict = rfc.predict(X_train)

    print('\n')
    print("=== Trained Classification Report ===")
    print(classification_report(y_train, rfc_training_predict))
    print('\n')


def create_training(image_input_dir, geo_data_dir, output_dir, version="0", window_size=3000, padding=20, resolution=3, percentage_sample=1):
    filenames = os.listdir(Path(geo_data_dir))
    training_dir = Path(output_dir).joinpath(
        "training").joinpath("version_{}".format(version))
    training_dir.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool()
    for filename in filenames[0:5]:
        image_input_src = Path(image_input_dir).joinpath(
            filename.replace('geojson', 'tif'))
        geo_data_src = Path(geo_data_dir).joinpath(filename)

        training_set = create_training_set_single_async(
            output_dir, image_input_src, geo_data_src, window_size, padding, resolution, percentage_sample, pool)
        training_set.to_csv(training_dir.joinpath(
            filename.replace('geojson', 'csv')))
        print("Completed: {}".format(filename.replace('.geojson', '')))

    pool.close()
    pool.join()


def create_training_set_single_async(output_dir, image_input_src, geo_data_src, window_size, padding, resolution, percentage_sample, pool):
    dataset = rasterio.open(image_input_src)

    training_set_ls = []

    with open(geo_data_src) as f:
        geo_data = json.load(f)
    labels = create_label(dataset, geo_data)

    def callback(training_set):
        training_set_ls.append(training_set)

    def error_callback(error):
        print(str(error))

    results = []
    for y in range(padding, dataset.height - padding, window_size):
        for x in range(padding, dataset.width - padding, window_size):
            result = pool.apply_async(create_training_set_section, args=(
                image_input_src, geo_data_src, labels, [y, x], window_size, padding, resolution, percentage_sample), error_callback=error_callback, callback=callback)
            results.append(result)

    [result.wait() for result in results]
    full_training_set = pd.DataFrame()
    for training_set in training_set_ls:
        full_training_set = full_training_set.append(training_set)

    return full_training_set


def create_training_set_section(image_input_src, geo_data_src, labels, offset, window_size, padding, resolution, percentage_sample):
    dataset = rasterio.open(image_input_src)
    image_data = load_window(dataset, offset, window_size, padding)
    height, width = image_data[padding:-
                               padding: resolution, padding: -padding: resolution, 0].shape
    training_set = extract_features(image_data, resolution, padding)
    training_set['label'] = labels[offset[0]: offset[0] +
                                   height, offset[1]: offset[1] + width].flatten()

    positive_set = training_set[training_set.label == False]
    negative_set = training_set[training_set.label == True]

    n_samples = (height * width * percentage_sample) / 200
    positive_frac = (n_samples / len(positive_set)
                     ) if len(positive_set) > n_samples else 1
    negative_frac = (n_samples / len(negative_set)
                     ) if len(negative_set) > n_samples else 1

    final_training_set = pd.DataFrame()
    final_training_set = final_training_set.append(
        positive_set.sample(frac=positive_frac))
    final_training_set = final_training_set.append(
        negative_set.sample(frac=negative_frac))

    return final_training_set


def create_label(dataset, geo_data):
    shapes = []
    for feature in geo_data['features']:
        shapes.append(geometry.Polygon(
            [[p[0], p[1]] for p in feature['geometry']['coordinates'][0]]))
    # create mask from shapes
    return rasterio.mask.raster_geometry_mask(dataset, shapes)[0]
