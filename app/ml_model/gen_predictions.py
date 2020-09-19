import netCDF4
import numpy as np
import rasterio
from shapely.geometry import Point

from ml_model import map_utils, predict, xml_parser


class MapPredictor:
    def __init__(self, area: int, viirs_data: str, rr_data: netCDF4.Dataset, evc_path: str, slp_path: str,
                 firecast_path: str):
        self.mu = map_utils.MapUtils(area)
        self.mu.reproj_viirs(viirs_data)
        self.rr_vars = ['tmp2m', 'rh2m', 'ugrd10m', 'vgrd10m', 'pratesfc']
        self.rr_data = {var: rr_data[var] for var in self.rr_vars}
        self.evc_path = evc_path
        self.evc_vars = ['Sparse Vegetation Canopy', 'Tree Cover', 'Shrub Cover', 'Herb Cover']
        self.evc_dict = xml_parser.parse_layers(evc_path + '.aux.xml')
        self.slp_path = slp_path
        self.area = area
        self.predictor = predict.Predictor(firecast_path)

    def gen_predictions(self) -> [(Point, np.float)]:
        """
        Calculate all predictions in the given VIIRS GeoDataFrame.
        """
        perimeters = self.mu.generate_perimeters()
        rr_x, rr_y = (0, 0)
        predictions = []
        with rasterio.Env():
            slp_tiff = rasterio.open(self.slp_path)
            evc_tiff = rasterio.open(self.evc_path)
            for p in perimeters:
                if self.mu.rr_indices(p[0].x, p[0].y) != [rr_x, rr_y]:
                    weather = np.zeros(len(self.rr_vars))
                    rr_x, rr_y = self.mu.rr_indices(p[0].x, p[0].y)
                    for i, var in enumerate(self.rr_vars):
                        weather[i] = self.rr_data[var][-1, rr_x, rr_y]
                    weather = np.expand_dims(weather, axis=0)
                slp = self.mu.read_tiff(slp_tiff, p[0].x, p[0].y)
                slp = self.mu.resample_landfire(slp)
                evc = self.mu.read_tiff(evc_tiff, p[0].x, p[0].y)
                evc_layers = []
                for var in self.evc_vars:
                    evc_var = sum([evc == i for i in self.evc_dict[var]])
                    evc_layers.append(self.mu.resample_landfire(evc_var))
                terrain = np.expand_dims(np.stack([p[1], slp] + evc_layers), axis=0)
                predictions.append((p[0], self.predictor.predict(terrain, weather)))
        return predictions
