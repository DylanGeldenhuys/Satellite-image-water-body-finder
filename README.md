# satalite-image-water-body-finder

Authors _S.R.K Gasson & D.S Geldenhuys_

satelite-image-water-body-finder is a computer vision system that detects and delineates water bodies using a random forest classifier at its core.

Setup on Windows:

Before installing the dependencies, it is required to install the windows binaries for the following:

-Gdal
-Rasterio
-Shapely

These can all be downloaded from the following link:
https://www.lfd.uci.edu/~gohlke/pythonlibs/
Use ctr+f to search through the windows binaries on this page and locate the wheels. Select the correct version for you system. The first in the list is typically the latest and correct version.

To install the wheels, run the following from within your environment for each wheel:

    pip install <path-to-wheel>

For example:

    pip install C:\wheels\GDAL-3.0.4-cp38-cp38-win_amd64.whl
    pip install C:\wheels\rasterio-1.1.3-cp38-cp38-win_amd64.whl
    pip install C:\wheels\Shapely-1.7.0-cp38-cp38-win_amd64.whl

Once these wheels are installed, run the following from within the root of the cloned repository to install dependencies and the water_body_finder package:

    pip install .

See the demo.py file for an example of how to use the water_body_finder package.
