import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from firecast import FireCast
from wildfire_data_loader import WildfireDataset

DATASET_PATH = './wildfire_data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train_v2-002.hdf5'
trainset = WildfireDataset(DATASET_PATH)

norm = transforms.Normalize(trainset.mean, trainset.std)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)

area = 15
perim = (area - 1) // 2
net = FireCast(area, terrain_features=trainset.terrain_features, weather_features=trainset.weather_features)

lr = 0.0001
criterion = nn.BCELoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr)
epochs = 2

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        running_loss = 0.0
        data = norm(data[0]).reshape(data.shape)
        pad_width = ((0, 0), (0, 0), (perim, perim), (perim, perim))
        terrain_full = np.pad(data[:, :trainset.terrain_features], pad_width)
        weather_full = data[:, trainset.terrain_features:-1]
        target = data[:, -1]

        # output = np.zeros(target.shape)
        for x in range(target.shape[1]):
            for y in range(target.shape[2]):
                terrain = np.zeros((1, 6, area, area))
                for layer in range(6):
                    terrain[0][layer] = terrain_full[0, layer, x:x + area, y:y + area]
                weather = weather_full[0:1, :, x, y]
                terrain = torch.tensor(terrain)
                pred = net(terrain, weather)
                # output[:, x, y] = pred[0][0].item()
                optimizer.zero_grad()
                loss = criterion(pred[0].double(), target[:, x, y])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        # print after every example
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (target.shape[1] * target.shape[2]):.3f}')

print('Finished Training')

firecast_path = './firecast.pth'
torch.save(net, firecast_path)

transform_path = './transform.pth'
torch.save(norm, transform_path)
