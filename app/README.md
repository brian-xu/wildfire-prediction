# End User Application

The files in this directory are a rough approximation of how I would have structured
a production-level deployment of the model. The data processing files [(ml_model/)](ml_model)
were written with scalability and optimization in mind, while the actual visualization
website [(static/](static) [and templates/)](templates) are more simplistic.

My implementation lends itself well to an architecture where a central server performs all
predictions at regular intervals, and uploads the resulting predictions to a central location.
The user application merely retrieves and displays this data, in this case using
[Leaflet.heat](https://github.com/Leaflet/Leaflet.heat).

# Data Processing

As-is, the code expects the trained model to be located at [ml_model/weights/firecast.pth](ml_model/weights/firecast.pth),
and all .tiff and .xml files genereated by GDAL to be located under [ml_model/maps](ml_model/maps).
Most of the functions are tailored to return data suitable for a model as defined in this
repository. However, many of the functions in [data.py](ml_model/data.py) and [map_utils.py](ml_model/map_utils.py)
are not application-specific and useful for manipulation of their respective datasets for any
purpose.

[gen_predictions](ml_model/gen_predictions.py) and [gen_heat](ml_model/gen_heat.py)
are the main files that would compute predictions on a regular interval.

# Data Visualization

The visualization demo I wrote was written simply in Leaflet and uses Leaflet.heat
as an efficient way to aggregate and display predictions. It requires no direct
connection to the server, instead importing the predictions from a hosted javascript
file.