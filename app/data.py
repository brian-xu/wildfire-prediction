import datetime
import io

import cv2
import netCDF4
import numpy as np
import requests


class DataCollector:
    def __init__(self, auth_key):
        self.date = datetime.date.today()
        self.auth_key = auth_key

    def viirs_data(self, date=None):
        if date is None:
            date = self.date
        viirs_url = 'https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/' \
                    'FIRMS/suomi-npp-viirs-c2/USA_contiguous_and_Hawaii/' \
                    'SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_VNP14IMGTDL_NRT_' + \
                    date.strftime('%Y%j') + '.txt'
        viirs_data = requests.get(viirs_url,
                                  headers={'Authorization': f'Bearer {self.auth_key}'}).text
        return viirs_data

    def landfire_data(self, bbox):
        landfire_url = 'https://landfire.cr.usgs.gov/arcgis/rest/services/Landfire/US_140/MapServer/export'
        landfire_data = requests.get(landfire_url,
                                     params={'bbox': bbox,
                                             'dynamicLayers': '[{"id":0,"source":{"type":"mapLayer","mapLayerId":"8"}}]',
                                             'size': '30,30',
                                             'format': 'png',
                                             'transparent': 'true',
                                             'f': 'image'})
        byte_stream = io.BytesIO(landfire_data.content)
        image = cv2.imdecode(np.frombuffer(byte_stream.read(), np.uint8), 1)
        return image

    def rapid_refresh_data(self, date=None):
        if date is None:
            date = self.date
        rapid_refresh_url = 'https://nomads.ncep.noaa.gov:9090/dods/rap/rap' + date.strftime("%Y%m%d") + '/rap_00z'
        with netCDF4.Dataset(rapid_refresh_url) as file:
            rr_vars = ['tmp2m', 'rh2m', 'ugrd10m', 'vgrd10m', 'pratesfc']
            rr_data = [file.variables[var][:] for var in rr_vars]
        return rr_data
