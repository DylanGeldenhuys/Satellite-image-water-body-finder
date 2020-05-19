from water_body_finder import find_waterbodies, create_training

if __name__ == '__main__':

    create_training("/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs",
                    "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines", "/media/ds/New Volume/Waterbody_Project/TESTING/training_output", window_size=900)

    #find_waterbodies("/media/ds/New Volume/Waterbody_Project/Test_images1/test_image", #input
    #                 "/media/ds/New Volume/Waterbody_Project/Test_images1/output", padding=1200, window_size=3000) #output

    