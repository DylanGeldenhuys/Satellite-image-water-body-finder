from water_body_finder import find_waterbodies, create_training

if __name__ == '__main__':

    # create_training("D:/WaterBodyExtraction/WaterPolyData/image_data",
    #                "D:/WaterBodyExtraction/WaterPolyData/geo_data/v2", "D:/Demo/output")

    find_waterbodies("D:/Demo/input",
                     "D:/Demo/output", padding=1200, window_size=3000)
