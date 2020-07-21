from data import DataCollector
from ml_model.gen_predictions import MapPredictor

dc = DataCollector(open('lance_key.txt').read())
mp = MapPredictor(dc.viirs_data(), dc.rapid_refresh_data(), 'ml_model/maps/evc.tiff',
                  'ml_model/maps/slp.tiff', 15, 'ml_model/weights/firecast.pth')
points = []
for i in mp.viirs_generator():
    for p in mp.gen_predictions([i]):
        if round(p[1], 2) > 0:
            points.append([p[0].x, p[0].y, round(p[1], 2)])

with open("static/latest.js", "w") as f:
    f.write(f'var firePoints = {points};')
