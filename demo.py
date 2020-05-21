from water_body_finder import find_waterbodies, create_training, train_model, save_prediction_mask

if __name__ == '__main__':

    # create_training("D:/WaterBodyExtraction/WaterPolyData/image_data",
    #                "D:/WaterBodyExtraction/WaterPolyData/geo_data/v2", "D:/Demo/output", ver)

    train_model("D:/Demo/output", "D:/Demo/output/training",
                version=0, sub_version=1)
    # save_prediction_mask(
    #   "D:/WaterBodyExtraction/WaterPolyData/image_data", "D:/Demo/output", "D:/Demo/output/rfc", "0_1", 50, 3000, 3)

    # find_waterbodies("D:/WaterBodyExtraction/WaterPolyData/image_data",
    #                 "D:/Demo/output", padding=1200, window_size=3000)
