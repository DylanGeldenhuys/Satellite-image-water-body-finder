if __name__ == '__main__':  # noqa

    import sys
    sys.path.append('C:/Users/SimonGasson/Documents/Projects/satalite-image-water-body-finder/data_processing/rfc_processing')  # noqa

    import os
    import multiprocessing as mp
    from create_test_set import create_test_set
    from pathlib import Path
    import pickle

    # define parameters
    image_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/image_data")
    geo_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/geo_data/v1")
    label_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/label_data/v1")

    training_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/testing_sets/set_9")
    visualisation_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/testing_sets/set_9/visualisations")

    pool = mp.Pool(mp.cpu_count())

    filenames = [
        "2531BD19",
        "2628BB25",
        "3319DC01",
        "3419BC03",
        "2531DA03",
        "2931AD12",
        "2828BA09",
        "2828BA09"
    ]
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
        pool.apply_async(create_test_set, args=(
            filename, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory), callback=collect_result)

    pool.close()
    pool.join()

    print(results)
    print("Completed")
