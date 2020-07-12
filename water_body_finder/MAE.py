# Imports

from pathlib import Path
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from geojson import Point, Feature, FeatureCollection, dump





# Directories
ref_geojsons = Path("D:\Waterbody_Project\FINAL\TRAINING\Polygons")
pred_geojsons = Path("D:\Waterbody_Project\FINAL\geo_data")
output = Path('.')


images = Path("Path/to/directory")


# functions
def MAEi(extracted_boundary, reference_boundary):
    '''measures the distance from each reference boundary (50x50cm) cell to the nearest extracted
    boundary cell, which provides an indication of how close in geographical space the extracted
    boundaries are to the actual boundary.'''

    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree').fit(extracted_boundary)
    distances = nbrs.kneighbors(reference_boundary)
    return(np.average(distances[0]))


def MAEj(reference_boundary, extracted_boundary):
    '''MAEj measures the distance from each extracted boundary cell to the nearest reference boundary
    cells and is an indicator of boundaries within a water body (caused by oversemgentation)'''

    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree').fit(reference_boundary)
    distances = nbrs.kneighbors(extracted_boundary)
    return(np.average(distances[0]))



def get_MAE(ref_geojsons,pred_geojsons, output):
    data_results = []
    MAEi_features= []
    MAEj_features = [] 
    ref_geojsons = Path(ref_geojsons)
    pred_geojsons = Path(pred_geojsons)
    output = Path(output)
    for filename in os.listdir(pred_geojsons)[5:6]:
        print(ref_geojsons.joinpath(filename))
        ref_list = []
        pred_list = []
        ref_poly = []
        pred_poly = []
        geo_ref_file = ref_geojsons.joinpath(filename)
        geo_pred_file = pred_geojsons.joinpath(filename)
        with open(geo_ref_file,'r' ) as f:
            geo_ref = json.load(f)
        with open(geo_pred_file, 'r') as f:
            geo_pred = json.load(f)
        # complete full list (list) of coordinates for prediction points   
        for feature in geo_pred['features']:
            for coordinate_group in feature['geometry']['coordinates']:
                while len(coordinate_group) == 1:
                    coordinate_group = coordinate_group[0]
                for p in coordinate_group:
                    pred_list.append([p[0], p[1]])

        # complete full list (list) of coordinates for refrence points
        for feature in geo_ref['features']:
            for coordinate_group in feature['geometry']['coordinates']:
                while len(coordinate_group) == 1:
                    coordinate_group = coordinate_group[0]
                for p in coordinate_group:
                    ref_list.append([p[0], p[1]])
        

        # create list of features (list of lists) for prediction points
        for feature in geo_pred['features']:
            for coordinate_group in feature['geometry']['coordinates']:
                while len(coordinate_group) == 1:
                    coordinate_group = coordinate_group[0]
                poly = [[p[0], p[1]] for p in coordinate_group]
                pred_poly.append(poly)
                MAEi_feat = MAEi(ref_list, poly)
                feature['properties'] = {'MAEi': MAEi_feat}

        with open(geo_pred_file, 'w') as f:
            json.dump(geo_pred,f)


        # create a list of features (list of lists) for refrence points
        for feature in geo_ref['features']:
            for coordinate_group in feature['geometry']['coordinates']:
                while len(coordinate_group) == 1:
                    coordinate_group = coordinate_group[0]
                poly = [[p[0], p[1]] for p in coordinate_group]
                ref_poly.append(poly)
                MAEi_feat = MAEi(pred_list, poly)
                feature['properties'] = {'MAEi': MAEi_feat}

        with open(geo_ref_file, 'w') as f:
            json.dump(geo_ref,f)

        # append file list
        data_results.append([filename,MAEi(pred_list, ref_list), MAEi(ref_list, pred_list)])
    
        for feature in ref_poly:
            MAEi_sample = MAEi(pred_list, feature)
            MAEi_features.append([filename, MAEi_sample])
        for feature in pred_poly:
            MAEi_sample = MAEi(ref_list, feature)
            MAEj_features.append([filename, MAEi_sample])
    datai = pd.DataFrame(MAEi_features)
    dataj = pd.DataFrame(MAEj_features)
    FINAL = pd.concat([datai,dataj], axis=1)
    FINAL.columns = ['filename_ref','MAEi', 'filename_pred', 'MAEi' ]
    FINAL.to_csv(output.joinpath('MAE.csv'))


    # print relevant stats
    data_results = pd.DataFrame(data_results, columns=['filename', 'MAEi', 'MAEj'])
    print(data_results)
    print('average MAEi:{}'.format(np.average(datai.iloc[:,1])) , 'average MAEj:{}'.format(np.average(dataj.iloc[:,1])) )

    # Print Histograms
    plt.figure(figsize=(10,10))
    plt.hist(datai.iloc[:,1])
    plt.title('MAEi')
    plt.xlabel('MAEi (m)')
    plt.ylabel('frequency')
    plt.savefig(output.joinpath('MAEi.png'))

    plt.figure(figsize=(10,10))
    plt.hist(dataj.iloc[:,1])
    plt.title('MAEj')
    plt.xlabel('MAEi (m)')
    plt.ylabel('frequency')
    plt.savefig(output.joinpath('MAEj.png'))

#if __name__ == "__main__":
#    get_MAE(ref_geojsons, pred_geojsons,output )


        









