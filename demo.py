from water_body_finder import find_waterbodies, create_training, train_model

if __name__ == '__main__':

    # create_training("D:/WaterBodyExtraction/WaterPolyData/image_data",
    #                "D:/WaterBodyExtraction/WaterPolyData/geo_data/v2", "D:/Demo/output")

    #train_model("D:/Demo/output", "D:/Demo/output/training")

    find_waterbodies("D:/WaterBodyExtraction/WaterPolyData/image_data",
                     "D:/Demo/output", padding=1200, window_size=3000)
