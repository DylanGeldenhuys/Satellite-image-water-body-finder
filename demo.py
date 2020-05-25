from water_body_finder import find_waterbodies, create_training, train_model, save_prediction_mask, save_polygons

if __name__ == '__main__':

    find_waterbodies(image_input_dir="D:/WaterBodyExtraction/WaterPolyData/image_data",
                     output_dir="D:/Demo/output", padding=50, window_size=3000)

    # Training
    # create_training(image_input_dir="D:/WaterBodyExtraction/WaterPolyData/image_data",
    #                 geo_data_dir="D:/WaterBodyExtraction/WaterPolyData/geo_data/v2", output_dir="D:/Demo/output", percentage_sample=2)

    # train_model(output_dir="D:/Demo/output", training_dir="D:/Demo/output/training",
    #            version=0, sub_version=0)

    # Testing
    # save_prediction_mask(
    #    image_input_dir="D:/WaterBodyExtraction/WaterPolyData/image_data", output_dir="D:/Demo/output",
    #   rfc_dir="D:/Demo/output/rfc", version="1", padding=50, window_size=3000, resolution=3)

    # save_polygons(
    #     image_input_dir="D:/WaterBodyExtraction/WaterPolyData/image_data", output_dir="D:/Demo/output",
    #     rfc_dir="C:/Users/SimonGasson/Documents/Projects/satalite-image-water-body-finder/water_body_finder/rfc", version="1", padding=50, window_size=3000, resolution=3)
