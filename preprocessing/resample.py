import math

import rasterio
from rasterio.transform import Affine

with rasterio.Env():
    evc = rasterio.open('evc.tiff')

    scale_factor = 12.5

    transform = evc.transform * Affine.scale(scale_factor)

    evc_scaled = rasterio.open(
        'evc_scaled.tiff',
        'w',
        driver='GTiff',
        height=math.ceil(evc.height / scale_factor),
        width=math.ceil(evc.width / scale_factor),
        count=evc.count,
        dtype=evc.dtypes[0],
        crs=evc.crs,
        transform=transform
    )

    evc_scaled.close()
