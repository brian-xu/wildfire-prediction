# LANDFIRE Preprocessing

LANDFIRE data is available to download from the [LANDFIRE website](https://www.landfire.gov/version_comparison.php).
The data is downloaded as zipped ARCGIS files, which can be translated to GeoTIFF files that are more
easily manipulated using GDAL.

[resample.py](resample.py) automates the resampling process of. The full-res maps
are about 30GB each, motivating resampling before rather than during usage. Such a large
file could bottleneck performance and take up unnecessary amounts of space.