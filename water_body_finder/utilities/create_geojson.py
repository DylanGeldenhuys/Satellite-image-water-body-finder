# Author: Dylan Geldenhuys

from shapely.wkt import loads
from shapely.geometry import mapping
from geojson import Point, Feature, FeatureCollection, dump
import rasterio
from shapely.geometry import LineString



def pixelcoord_to_geocoord(pixel_coordinate):
    #pixel coordinate must be tuple type
    # define raterio_object outside the function, rasterio_object = rasterio.open(image)
    return(rasterio_object.transform * pixel_coordinate)


def create_geojson(ordered_list, rasterio_object,image_name):
    #rasterio_object = rasterio.open(geoTif)
    features = []
    for l in ordered_list[:4]:
        tuple_of_tuples = tuple(tuple(x) for x in l)
        Lstring = LineString(list(map(pixelcoord_to_geocoord,tuple_of_tuples)))
        features.append(Feature(geometry=Lstring))
    feature_collection = FeatureCollection(features)
    with open('{}.geojson'.format(image_name), 'w') as f:
        dump(feature_collection, f)
