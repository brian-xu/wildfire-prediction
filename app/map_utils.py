import math

import pyproj

m_transformers = []
d_transformers = []

for x in range(10):
    m_transformers.append(pyproj.Transformer.from_crs("epsg:4326", f"epsg:{6339 + x}"))
    d_transformers.append(pyproj.Transformer.from_crs(f"epsg:{6339 + x}", "epsg:4326"))


def utm_offset(longitude: float) -> int:
    return math.floor((longitude + 126) / 6)


class MapUtils:
    def __init__(self):
        self.to_m = m_transformers
        self.to_deg = d_transformers

    def bbox(self, latitude: float, longitude: float, area: int) -> [float]:
        zone = utm_offset(longitude)
        x, y = self.to_m[zone].transform(latitude, longitude)
        x_min, y_min = self.to_deg[zone].transform(x - (area / 2 * 375), y - (area / 2 * 375))
        x_max, y_max = self.to_deg[zone].transform(x + (area / 2 * 375), y + (area / 2 * 375))
        return [x_min, y_min, x_max, y_max]

    def rr_indexes(self, latitude: float, longitude: float):
        lat_index = (latitude - 16.32201100000) // 0.18470909
        lon_index = (longitude + 139.85660300000) // 0.19242568
        return [lat_index, lon_index]
