# Model Training

The model I based this project on implements the FireCast architecture as described
in the paper by [[Radke *et al.*, 2019]](https://www.ijcai.org/Proceedings/2019/636).

A certain idiosyncrasy of the model is it generates only one prediction for a given
input: as a result, it requires many predictions to be done in a perimeter around
each fire.

To train your own copy of the model as described in [firecast.py](firecast.py),
you should configure the [data loader](wildfire_data_loader.py) to your own
specifications. The dataset and visualization notebooks (availability pending)
should give you a clearer picture of the data used, and how to select features
for your own model. In particular, this line governs which LANDFIRE attributes
are used:

```python
landfire_attrs = ('SLP', 'Sparse', 'Tree', 'Shrub', 'Herb')
```

[train.py](train.py) and the model architecture should be flexible to changes
within reason, and run smoothly regardless of modifications made to the data loader.