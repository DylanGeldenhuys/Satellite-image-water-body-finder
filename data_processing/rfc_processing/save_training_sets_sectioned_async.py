if __name__ == '__main__':  # noqa

    import sys
    sys.path.append('C:/personal/satalite-image-water-body-finder/data_processing/rfc_processing')  # noqa

    import os
    import multiprocessing as mp
    from create_training_set_sectioned import create_training_set
    from pathlib import Path
    import pickle

    # define parameters
    image_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/image_data")
    label_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/label_data")

    training_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_7")
    visualisation_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/visualisations/training_set_7")

    image_label_fix_src = Path(
        "D:/WaterBodyExtraction/WaterPolyData/image_data_preprocess/label_fix.p")

    pool = mp.Pool(mp.cpu_count())

    results = []
    length = len(os.listdir(image_data_directory))

    def collect_result(result):
        global results
        results.append(result)
        print("{0} files of {1} complete".format(len(results), length))

    # load image label fix
    image_label_fix_data = pickle.load(open(image_label_fix_src, 'rb'))
    for i in range(10):
        if i in image_label_fix_data and len(image_label_fix_data[i]) > 0:
            pool.apply_async(create_training_set, args=(
                i, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory, image_label_fix_data), callback=collect_result)

    pool.close()
    pool.join()

    print(results)
    print("Completed")
