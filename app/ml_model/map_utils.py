import io
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
from rasterio.windows import Window
from shapely.geometry import Point

aea = CRS.from_user_input("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 "
                          "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True")
landfire = CRS.from_string('PROJCS["unnamed",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                           'SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],'
                           'AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],'
                           'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                           'AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],'
                           'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],'
                           'PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],'
                           'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
                           'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
                           'AXIS["Northing",NORTH]]')
deg_landfire = Transformer.from_crs("epsg:4326", landfire)

top = 49.3457868  # north lat
left = -124.7844079  # west lon
right = -66.9513812  # east lon
bottom = 24.7433195  # south lat


def rr_indices(lat: float, lon: float):
    lat_index = (lat - 16.3220110) // 0.18470909
    lon_index = (lon + 139.8566030) // 0.19242568
    return [lat_index, lon_index]


def reproj_viirs(viirs_data: str):
    df = pd.read_csv(io.StringIO(viirs_data))
    df = df[df.latitude >= bottom]
    df = df[df.latitude <= top]
    df = df[df.longitude >= left]
    df = df[df.longitude <= right]
    return gpd.GeoDataFrame(df, crs="EPSG:4326",
                            geometry=gpd.points_from_xy(df.longitude, df.latitude)
                            ).to_crs(crs=aea).iloc[:, [0, 1, -1]]


def generate_perimeters(gdf, index, area):
    perimeters = []
    fire_center = gdf.iloc[index, :]
    center_lon, center_lat = fire_center.geometry.x, fire_center.geometry.y
    gdf = gdf[gdf.geometry.x >= center_lon - area * 375]
    gdf = gdf[gdf.geometry.x <= center_lon + area * 375]
    gdf = gdf[gdf.geometry.y >= center_lat - area * 375]
    gdf = gdf[gdf.geometry.y <= center_lat + area * 375]
    perim = (area - 1) // 2
    for lat in range(-perim, perim + 1):
        for lon in range(-perim, perim + 1):
            top_left = Point(center_lon + (lon - perim - 0.5) * 375, center_lat + (lat - perim - 0.5) * 375)
            center = Point(fire_center.longitude, fire_center.latitude)
            fire_data = np.zeros((area, area))
            for index, fire in gdf.iterrows():
                array_x = int((fire.geometry.x - top_left.x) // 375)
                array_y = int((fire.geometry.y - top_left.y) // 375)
                if 0 <= array_x < area and 0 <= array_y < area:
                    fire_data[array_y, array_x] = 1
            perimeters.append((center, fire_data))
    return perimeters


def read_tiff(tiff, lat, lon, area):
    perim = (area - 1) // 2
    center_lon, center_lat = deg_landfire.transform(lat, lon)
    y, x = tiff.index(center_lon - (perim + 0.5) * 375, center_lat + (perim + 0.5) * 375)
    data = tiff.read(1, window=Window(x, y, math.ceil(area * 12.5), math.ceil(area * 12.5)))
    return data


def resample_landfire(landfire, area):
    resampled = np.zeros((area, area))
    pool_mask = np.ones_like(landfire, dtype='float32')
    lf_area = landfire.shape[0]
    for x in range(12, lf_area, 25):
        pool_mask[:, x] *= 0.5
        pool_mask[x, :] *= 0.5
    for x in range(math.floor(lf_area / 12.5)):
        for y in range(math.floor(lf_area / 12.5)):
            chunk_x = math.floor(12.5 * x)
            chunk_y = math.floor(12.5 * y)
            chunk = landfire[chunk_x:chunk_x + 13, chunk_y:chunk_y + 13]
            chunk_mask = pool_mask[chunk_x:chunk_x + 13, chunk_y:chunk_y + 13]
            resampled[x, y] = np.average(chunk, weights=chunk_mask)
    return resampled
