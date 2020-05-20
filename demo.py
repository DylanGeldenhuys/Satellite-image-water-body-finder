from water_body_finder import find_waterbodies, create_training, train_model

if __name__ == '__main__':

    create_training("/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs",
                    "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines", "/media/ds/New Volume/Waterbody_Project/TESTING/training_output", window_size=900)

<<<<<<< HEAD
    #find_waterbodies("/media/ds/New Volume/Waterbody_Project/Test_images1/test_image", #input
    #                 "/media/ds/New Volume/Waterbody_Project/Test_images1/output", padding=1200, window_size=3000) #output

    
=======
    #train_model("D:/Demo/output", "D:/Demo/output/training")

    find_waterbodies("D:/WaterBodyExtraction/WaterPolyData/image_data",
                     "D:/Demo/output", padding=1200, window_size=3000)
>>>>>>> e96ab9e441807a034e9878bba145f5b75f3644f9
