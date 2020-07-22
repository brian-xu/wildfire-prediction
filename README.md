# Wildfire Prediction

This is a complete machine learning application for predicting wildfire spread
in the United States. The model architecture is written in PyTorch, with heavy
use of GeoPandas and rasterio for data processing.

# Demos

The purpose of this project was mainly self-educational, and being a student I
decided not to invest the considerable amount of resources training and
deploying a production-quality model would require. As a result, a permanently
running and public demo was out of the scope of the project (although the
architecture would lend itself well to a GitHub pages one.) Nonetheless,
I wrote scalable, efficient, and accurate code for training and deployment
and verified convergence on a subset of the training data.

[app/static](app/static) contains a very lightweight JavaScript demo for
displaying results.

# Reproduction

To the best of my ability, I have described the necessary steps to configure
and deploy an application using all or some of the code in this repository.
Each folder has a README with specifics for that aspect of the project.

Under the [BSD 3 License](LICENSE), you are free to distribute and modify the
contents of this repository to your liking.

# Credits and Further Reading

Training data and visualization notebooks provided by Casey Graff.

Original FireCast architecture and algorithm described in [[Radke *et al.*, 2019]](https://www.ijcai.org/Proceedings/2019/636).

Mapping demo uses [Leaflet](https://github.com/Leaflet/Leaflet) and [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat).