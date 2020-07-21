import io
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
from rasterio.windows import Window
from shapely.geometry import Point


class MapUtils:
    def __init__(self, area):
        self.area = area
        self.radius = (area - 1) // 2
        self.aea = CRS.from_user_input("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 "
                                       "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True")
        self.landfire = CRS.from_string('PROJCS["unnamed",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                                        'SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],'
                                        'AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],'
                                        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                                        'AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],'
                                        'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],'
                                        'PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],'
                                        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
                                        'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
                                        'AXIS["Northing",NORTH]]')
        self.aea_deg = Transformer.from_crs(self.aea, "epsg:4326")
        self.deg_landfire = Transformer.from_crs("epsg:4326", self.landfire)
        self.gdf = None

        self.top = 49.3457868  # north lat
        self.left = -124.7844079  # west lon
        self.right = -66.9513812  # east lon
        self.bottom = 24.7433195  # south lat

    def rr_indices(self, lat: float, lon: float):
        lat_index = (lat - 16.3220110) // 0.18470909
        lon_index = (lon + 139.8566030) // 0.19242568
        return [lat_index, lon_index]

    def reproj_viirs(self, viirs_data: str):
        df = pd.read_csv(io.StringIO(viirs_data))
        df = df[df.latitude >= self.bottom]
        df = df[df.latitude <= self.top]
        df = df[df.longitude >= self.left]
        df = df[df.longitude <= self.right]
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326",
                               geometry=gpd.points_from_xy(df.longitude, df.latitude)
                               )
        self.gdf = gdf.to_crs(crs=self.aea).iloc[:, [0, 1, -1]]

    def generate_perimeters(self, index):
        perimeters = []
        fire_center = self.gdf.iloc[index, :]
        center_lon, center_lat = fire_center.geometry.x, fire_center.geometry.y
        gdf = self.gdf[self.gdf.geometry.x >= center_lon - self.area * 375]
        gdf = gdf[gdf.geometry.x <= center_lon + self.area * 375]
        gdf = gdf[gdf.geometry.y >= center_lat - self.area * 375]
        gdf = gdf[gdf.geometry.y <= center_lat + self.area * 375]
        gdf_array = np.zeros((self.area * 2 - 1, self.area * 2 - 1))
        top_left = Point(center_lon - (self.area - 0.5) * 375, center_lat - (self.area - 0.5) * 375)
        for index, fire in gdf.iterrows():
            array_x = int((fire.geometry.x - top_left.x) // 375)
            array_y = int((fire.geometry.y - top_left.y) // 375)
            if 0 <= array_x < self.area * 2 - 1 and 0 <= array_y < self.area * 2 - 1:
                gdf_array[array_y, array_x] = 1
        for lat in range(-self.radius, self.radius + 1):
            for lon in range(-self.radius, self.radius + 1):
                center = Point(self.aea_deg.transform(center_lon + lon * 375, center_lat + lat * 375))
                fire_data = gdf_array[self.radius + lat:self.area + self.radius + lat,
                            self.radius + lon:self.area + self.radius + lon]
                perimeters.append((center, fire_data))
        return perimeters

    def read_tiff(self, tiff, lat, lon):
        center_lon, center_lat = self.deg_landfire.transform(lat, lon)
        y, x = tiff.index(center_lon - (self.radius + 0.5) * 375, center_lat + (self.radius + 0.5) * 375)
        data = tiff.read(1, window=Window(x, y, math.ceil(self.area * 12.5), math.ceil(self.area * 12.5)))
        return data

    def resample_landfire(self, lf_data):
        lf_area = lf_data.shape[0]
        resampled = np.zeros((self.area, self.area))
        pool_mask = np.ones((lf_area, lf_area))
        for x in range(12, lf_area, 25):
            pool_mask[:, x] *= 0.5
            pool_mask[x, :] *= 0.5
        for x in range(math.floor(lf_area / 12.5)):
            for y in range(math.floor(lf_area / 12.5)):
                chunk_x = math.floor(12.5 * x)
                chunk_y = math.floor(12.5 * y)
                chunk = lf_data[chunk_x:chunk_x + 13, chunk_y:chunk_y + 13]
                chunk_mask = pool_mask[chunk_x:chunk_x + 13, chunk_y:chunk_y + 13]
                resampled[x, y] = np.average(chunk, weights=chunk_mask)
        return resampled
