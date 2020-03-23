import rasterio
from rasterio import mask
from shapely import geometry

def create_mask(image_data, geo_data):
    #Create shapes
    shapes = []
    for feature in geo_data['features']:
        shapes.append(geometry.Polygon([[p[0], p[1]] for p in feature['geometry']['coordinates']]))

    #Create mask
    return(rasterio.mask.raster_geometry_mask(image_data, shapes)[0])