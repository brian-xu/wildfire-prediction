# Model Training

The model implements the FireCast architecture as described in this paper by
[[Radke *et al.*, 2019].](https://www.ijcai.org/Proceedings/2019/636)

The model generates only one prediction for a given input as opposed to perhaps
a more generative approach: as a result, it requires many predictions for each fire.

To train your own copy of the model as described in [firecast.py](firecast.py),
you should configure the [data loader](wildfire_data_loader.py) to your own
specifications. The dataset and visualization notebooks (availability pending)
should give you a clearer picture of the data used, and how to select features
for your own model. In particular, this line governs which LANDFIRE attributes
are used:

```python
landfire_attrs = ('SLP', 'Sparse', 'Tree', 'Shrub', 'Herb')
```

The [training loop](train.py) and model architecture should account for any
changes made in the data loader, but I didn't test this extensively.