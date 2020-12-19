# LANDFIRE Preprocessing

LANDFIRE data is available to download from the [LANDFIRE website](https://www.landfire.gov/version_comparison.php)
as zipped ARCGIS files, which can be converted using GDAL to GeoTIFF files that are more
easily read.

The following GDAL command can be used to translate the unzipped directories:
```batch
gdal_translate -co COMPRESS=LZW -ot Int16 -of GTiff path_to_adf_files landfire.tiff
```

`path_to_adf_files` should be a directory containing unzipped adf files and look something
like `Slope\Grid\us_slp`.