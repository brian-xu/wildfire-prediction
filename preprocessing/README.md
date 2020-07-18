# LANDFIRE Preprocessing

LANDFIRE data is available to download from the [LANDFIRE website](https://www.landfire.gov/version_comparison.php).
The data is downloaded as zipped ARCGIS files, which can be translated using GDAL to GeoTIFF files that are more
easily manipulated.

The following GDAL command can be used to translate the unzipped directories:
```batch
gdal_translate -co COMPRESS=LZW -ot Int16 -of GTiff path_to_adf_files landfire.tiff
```

`path_to_adl_files` refers to the directory that contains the adf files and should look something like `Slope\Grid\us_slp`.