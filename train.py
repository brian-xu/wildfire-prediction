import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from firecast import FireCast
from wildfire_data_loader import WildfireDataset

DATASET_PATH = './wildfire_data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train_v2-002.hdf5'
trainset = WildfireDataset(DATASET_PATH)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)

area = 15
peri = (area - 1) // 2
net = FireCast(area, terrain_features=trainset.terrain_features, weather_features=trainset.weather_features)

lr = 0.0005
criterion = nn.BCELoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr)
epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        pad_width = ((0, 0), (0, 0), (peri, peri), (peri, peri))
        terrain_full = np.pad(data[:, :6], pad_width)
        weather_full = data[:, 6:-1]
        target = data[:, -1]

        batches = 0
        output = np.zeros(target.shape)
        for x in range(target.shape[1]):
            for y in range(target.shape[2]):
                batches += 1

                terrain = np.zeros((1, 6, area, area))
                for layer in range(6):
                    terrain[0][layer] = terrain_full[0, layer, x:x + area, y:y + area]
                weather = weather_full[0:1, :, x, y]
                terrain = torch.tensor(terrain)
                pred = net(terrain, weather)
                # output[:, x, y] = result[0][0].item()

                optimizer.zero_grad()
                loss = criterion(pred[0].double(), target[:, x, y])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batches % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

print('Finished Training')

PATH = './firecast.pth'
torch.save(net.state_dict(), PATH)
