class MapUtils:

    def __init__(self):
        pass

    # TODO: implement Albers equal-area projection in geopandas according to the following proj4 string:
    #  "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True"
    #  Implement function that creates numpy arrays of fire data in a perimeter around a VIIRS detection
    #  Create rasterio functions to scale, and pool areas of LANDFIRE data for inference
    #  Use the rr_indexes function to acquire RapidRefresh data at given points for inference

    def rr_indexes(self, latitude: float, longitude: float):
        lat_index = (latitude - 16.32201100000) // 0.18470909
        lon_index = (longitude + 139.85660300000) // 0.19242568
        return [lat_index, lon_index]
