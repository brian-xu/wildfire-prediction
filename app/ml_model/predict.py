import numpy as np
import torch


class Predictor:

    def __init__(self, firecast_path: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = torch.load(firecast_path, map_location=self.device)
        self.net.eval()

    def predict(self, terrain: np.array, weather: np.array) -> torch.tensor:
        with torch.no_grad():
            terrain = torch.tensor(terrain, device=self.device)
            weather = torch.tensor(weather, device=self.device)
            pred = self.net(terrain, weather)

        return pred[0].to('cpu').item()
