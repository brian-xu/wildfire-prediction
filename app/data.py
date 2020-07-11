import datetime

import netCDF4
import requests

today = datetime.date.today()
rapid_refresh_url = 'https://nomads.ncep.noaa.gov:9090/dods/rap/rap' + today.strftime("%Y%m%d") + '/rap_00z'
file = netCDF4.Dataset(rapid_refresh_url)

auth_key = open('lance_key.txt').read()

viirs_url = 'https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/' \
            'FIRMS/suomi-npp-viirs-c2/USA_contiguous_and_Hawaii/' \
            'SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_VNP14IMGTDL_NRT_' + \
            today.strftime('%Y%j') + '.txt'

print(requests.get(viirs_url,
                   headers={'Authorization': f'Bearer {auth_key}'}).text)

landfire_url = 'https://landfire.cr.usgs.gov/arcgis/rest/services/Landfire/US_140/MapServer/export'

print(requests.get(landfire_url,
                   params={'bbox': '-121.315,39.985,-121.165,40.135',
                           'format': 'png',
                           'transparent': 'false',
                           'dynamicLayers': '[{"id":0,"source":{"type":"mapLayer","mapLayerId":"8"}}]',
                           'f': 'pjson'}).text)
