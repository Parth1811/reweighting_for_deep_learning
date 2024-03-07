import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets

from model import myCNN
from alexNet import AlexNet

# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

train_dataset = datasets.CIFAR10("~/.torch/data", train=True, download=True, transform=AlexNet.get_input_transform(mu, cov))
train_loader = utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)

test_dataset = datasets.CIFAR10("~/.torch/data", train=False,download=True, transform=AlexNet.get_input_transform(mu, cov))
test_loader = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)

#cnn_model = myCNN()
cnn_model = AlexNet()
loss_fn = nn.CrossEntropyLoss()
#optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

device = torch.device("cpu")

cnn_model.train()
for epoch in range(10):

    loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        out = cnn_model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

    print(f"Running epoch: {epoch} | Loss: {loss.item()}")



cnn_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {correct / total * 100:.2f}%")
