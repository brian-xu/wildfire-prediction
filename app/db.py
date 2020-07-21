import datetime

import netCDF4
import requests


class DataCollector:
    def __init__(self, auth_key):
        self.date = datetime.date.today()
        self.auth_key = auth_key

    def viirs_data(self):
        viirs_url = 'https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/' \
                    'FIRMS/suomi-npp-viirs-c2/USA_contiguous_and_Hawaii/' \
                    'SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_VNP14IMGTDL_NRT_' + \
                    self.date.strftime('%Y%j') + '.txt'
        viirs_data = requests.get(viirs_url,
                                  headers={'Authorization': f'Bearer {self.auth_key}'}
                                  ).text
        return viirs_data

    def rapid_refresh_data(self):
        rapid_refresh_url = 'https://nomads.ncep.noaa.gov:9090/dods/rap/rap' + self.date.strftime("%Y%m%d") + '/rap_00z'
        return netCDF4.Dataset(rapid_refresh_url)

# TODO: Implement database class that stores and retrieves finished predictions
