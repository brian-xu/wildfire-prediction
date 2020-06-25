import torch
import torch.nn as nn
import torch.optim as optim
from numpy import np

from firecast import FireCast

area = 15
perim = (area - 1) // 2
net = FireCast(area)

lr = 0.005

criterion = nn.CrossEntropyLoss()
optimizer = optim.rmsprop.RMSprop(net.parameters(), lr=lr)

epochs = 10

trainloader = None

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        pad_width = ((0, 0), (0, 0), (perim, perim), (perim, perim))
        terrain_full = np.pad(data[:, :6], pad_width)
        weather_full = data[:, 6:-1]
        target = data[:, -1]

        optimizer.zero_grad()
        output = np.zeros(target.shape)
        for x in range(target.shape[0]):
            for y in range(target.shape[1]):
                terrain = np.zeros((1, 6, area, area))
                weather = np.zeros(5)
                for layer in range(6):
                    terrain[0][layer] = terrain_full[0][layer][x + perim:x + perim + area][y + perim:y + perim + area]
                weather = weather_full[0, :, x, y]
                output[x][y] = net(terrain, weather)[0].item()
        loss = criterion(torch.tensor(output), torch.tensor(target))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './firecast.pth'
torch.save(net.state_dict(), PATH)
