import h5py
import numpy as np

from torch.utils.data import Dataset


class WildfireDataset(Dataset):
    def __init__(self, dataset_path, normalize=False):
        LANDFIRE_INDICES = {'ASP': 0, 'CBD': 1, 'CBH': 2, 'CC': 3, 'CH': 4, 'DEM': 5, 'EVT': 6, 'SLP': 16,
                            'Nodata': 0, 'Sparse': 1, 'Tree': 2, 'Shrub': 3, 'Herb': 4, 'Water': 5, 'Barren': 6,
                            'Developed': 7, 'Snow-Ice': 8, 'Agriculture': 9}

        with h5py.File(dataset_path, 'r') as f:
            train_data = {}
            for k in list(f):
                train_data[k] = f[k][:]

        viirs_0 = train_data['observed'][:, 4:5]
        landfire_attrs = ('SLP', 'Sparse', 'Tree', 'Shrub', 'Herb')
        landfire = train_data['land_cover'][:, [LANDFIRE_INDICES[attr] for attr in landfire_attrs]]
        rapid_refresh = train_data['meteorology'][:, 4]
        viirs_12 = train_data['target'][:, 0:1]
        self.data = np.nan_to_num(np.concatenate((viirs_0, landfire, rapid_refresh, viirs_12), axis=1))
        self.terrain_features = len(landfire_attrs) + 1
        self.weather_features = rapid_refresh.shape[1]
        if normalize:
            self.terrain_mean = np.mean(self.data[:, :self.terrain_features], axis=(0, 2, 3))
            self.terrain_std = np.std(self.data[:, :self.terrain_features], axis=(0, 2, 3))
            self.weather_mean = np.mean(self.data[:, self.terrain_features:-1], axis=(0, 2, 3))
            self.weather_std = np.std(self.data[:, self.terrain_features:-1], axis=(0, 2, 3))
            self.terrain_mean[[0]] = 0
            self.terrain_std[[0]] = 1
        else:
            self.terrain_mean = np.zeros(self.terrain_features)
            self.terrain_std = np.ones(self.terrain_features)
            self.weather_mean = np.zeros(self.weather_features)
            self.weather_std = np.ones(self.weather_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
