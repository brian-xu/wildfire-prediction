from ml_model.data import DataCollector
from ml_model.gen_predictions import MapPredictor

dc = DataCollector(open('../lance_key.txt').read())
mp = MapPredictor(15, dc.viirs_data(), dc.rapid_refresh_data(), 'maps/evc.tiff',
                  'maps/slp.tiff', 'weights/firecast.pth')
points = []
for i in mp.viirs_generator():
    for p in mp.gen_predictions([i]):
        if round(p[1], 2) > 0:
            points.append([p[0].x, p[0].y, f'{p[1]*35:.4f}'])

with open("../static/latest.js", "w") as f:
    f.write(f'var firePoints = {points};')
