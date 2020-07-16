# TODO: Create rasterio functions to scale and pool areas of LANDFIRE data for inference
#  Use the rr_indexes function to acquire RapidRefresh data at given points for inference
import io

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj.crs import CRS
from shapely.geometry import Point

aea = CRS.from_user_input("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 "
                          "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True")

top = 49.3457868  # north lat
left = -124.7844079  # west lon
right = -66.9513812  # east lon
bottom = 24.7433195  # south lat


def rr_indexes(lat: float, lon: float):
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
    peri = (area - 1) // 2
    gdf.plot()
    plt.show()
    for lat in range(-peri, peri + 1):
        for lon in range(-peri, peri + 1):
            top_left = Point(center_lon + (lon - peri - 0.5) * 375, center_lat + (lat - peri - 0.5) * 375)
            center = Point(center_lon + lon * 375, center_lat + lat * 375)
            fire_data = np.zeros((area, area))
            for index, fire in gdf.iterrows():
                array_x = int((fire.geometry.x - top_left.x) // 375)
                array_y = int((fire.geometry.y - top_left.y) // 375)
                if 0 <= array_x < area and 0 <= array_y < area:
                    fire_data[array_y, array_x] = 1
            perimeters.append((center, fire_data))
    return perimeters
