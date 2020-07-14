import torch


class Predictor:

    def __init__(self, firecast_path, terrain_path, weather_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = torch.load(firecast_path, map_location=self.device)
        self.net.eval()
        self.terrain_norm = torch.load(terrain_path, map_location=self.device)
        self.weather_norm = torch.load(weather_path, map_location=self.device)

    def predict(self, terrain, weather) -> torch.tensor:
        with torch.no_grad():
            terrain = self.terrain_norm(torch.tensor(terrain))
            weather = self.weather_norm(torch.tensor(weather))
            pred = self.net(terrain, weather)

        return pred[0].to('cpu').item()
