if __name__ == '__main__':  # noqa

    import sys
    sys.path.append('/home/ds/Projects/satalite-image-water-body-finder/data_processing/rfc_processing')  # noqa

    import os
    import multiprocessing as mp
    from create_training_set_full import create_training_set
    from pathlib import Path
    import pickle

    # define parameters
    image_data_directory = Path(
        "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/tifs")
    geo_data_directory = Path(
        "/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines")
    label_data_directory = Path(
        "/media/ds/New Volume/Waterbody_Project/new_labels")

    training_output_directory = Path(
        "/media/ds/New Volume/Waterbody_Project/Training_set_1")
    visualisation_output_directory = Path(
        "/media/ds/New Volume/Waterbody_Project/visual_1")

    pool = mp.Pool(mp.cpu_count())

    filenames = os.listdir("/media/ds/New Volume/Waterbody_Project/raw_data/WaterPolyData/Polylines")
    results = []
    length = len(filenames)

    def collect_result(result):
        global results
        results.append(result)
        filename, success = result
        message = "Success" if success else "Failed"
        print("{0}: {1}".format(filename, message))
        print("{0} files of {1} complete".format(len(results), length))

    for filename in filenames:
        pool.apply_async(create_training_set, args=(
            filename.replace(".geojson", ""), image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory), callback=collect_result)

    pool.close()
    pool.join()

    print(results)
    print("Completed")
