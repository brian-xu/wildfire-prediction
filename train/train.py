import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from firecast import FireCast
from wildfire_data_loader import WildfireDataset

DATASET_PATH = 'wildfire_data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train_v2-002.hdf5'
trainset = WildfireDataset(DATASET_PATH)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, pin_memory=True)

area = 15
radius = (area - 1) // 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = FireCast(area, terrain_features=trainset.terrain_features, weather_features=trainset.weather_features).to(device)

lr = 0.0001
criterion = nn.BCELoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr)
epochs = 10

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        running_loss = 0.0
        data = data[0]
        target = data[-1]
        data = data[:-1]
        pad_width = ((0, 0), (radius, radius), (radius, radius))
        # terrain_full = torch.tensor(np.pad(data[:trainset.terrain_features], pad_width))
        terrain_full = data[:trainset.terrain_features]
        weather_full = data[trainset.terrain_features:]
        # output = np.zeros(target.shape)
        size = 0
        for x in range(radius, target.shape[0] - radius):
            for y in range(radius, target.shape[1] - radius):
                size += 1
                terrain = np.zeros((1, trainset.terrain_features, area, area))
                for layer in range(trainset.terrain_features):
                    terrain[0][layer] = terrain_full[layer, x - radius:x + radius + 1, y - radius:y + radius + 1]
                if np.sum(terrain[0][0]) > 0:
                    weather = weather_full[:, x, y].reshape((1, trainset.weather_features)).to(device)
                    terrain = torch.tensor(terrain).to(device)
                    pred = net(terrain, weather)
                    # output[x, y] = pred[0][0].item()
                    optimizer.zero_grad()
                    loss = criterion(pred[0].double(), target[x, y].reshape(1).to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

        # print after every example
        print(f'[{epoch + 1}, {i + 1}] loss:{running_loss / size :.3f}')

print('Finished Training')

firecast_path = './firecast.pth'
torch.save(net, firecast_path)
