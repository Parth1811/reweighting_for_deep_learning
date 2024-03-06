import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets

from model import myCNN

# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

train_dataset = datasets.CIFAR10("~/.torch/data", train=True, transform=myCNN.get_input_transform(mu, cov))
train_loader = utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)

test_dataset = datasets.CIFAR10("~/.torch/data", train=False, transform=myCNN.get_input_transform(mu, cov))
test_loader = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)

cnn_model = myCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cpu")

def compute_accuracy():
    cnn_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total * 100:.2f}%")


for epoch in range(30):

    loss = 0
    cnn_model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        out = cnn_model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

    print(f"Running epoch: {epoch} | Loss: {loss.item()}")

    if epoch % 5 == 0 and epoch != 0:
        compute_accuracy()

compute_accuracy()