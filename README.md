# Wildfire Prediction

This is a machine learning application for predicting wildfire spread.
The main architecture is written in PyTorch, and GeoPandas and rasterio
are used for data processing.

# Overview

I created this project with the intent of challenging myself to build a
complete machine learning application, from the model architecture to a
web application. I considered actually training and deploying a model, but
was unsatisfied with the 15-kilometer constraint imposed by the dataset.
Scraping historical data to create a new dataset was out of the scope of the
project, so I decided against it.

The repository is broken up into several components:

[train/](train) contains the model architecture and the training loop.

[preprocessing/](preprocessing) describes how to download and handle the
geographical data the model requires.

[app/](app) contains a script for generating fire spread predictions, as well
as code for an interactive map to view those predictions.

# Reproduction

Each folder has a README with specifics for that aspect of the project.

Under the [BSD 3 License](LICENSE), you are free to distribute and modify the
contents of this repository to your liking.

# Credits and Further Reading

Training data and visualization notebooks provided by Casey Graff.

Original FireCast architecture and algorithm described in [[Radke *et al.*, 2019]](https://www.ijcai.org/Proceedings/2019/636).

Mapping demo uses [Leaflet](https://github.com/Leaflet/Leaflet) and [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat).