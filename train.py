import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from model import myCNN
from dataloaders import ImbalancedCIFAR10Dataset

def stratified_split(dataset, test_size=0.1):
    labels = [label for _, label in dataset]
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=42,
        stratify=labels,
        shuffle=True)
    return train_idx, val_idx

def calculate_class_distribution(loader):
    class_counts = torch.zeros(10, dtype=torch.int64)
    for _, labels in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts

# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

train_dataset = datasets.CIFAR10("~/.torch/data", train=True, transform=myCNN.get_input_transform(mu, cov))

train_idx, val_idx = stratified_split(train_dataset, test_size=0.001)
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)

imbalanced_train_dataset = ImbalancedCIFAR10Dataset(train_subset, dominant_class=2, percentage=0.9)

train_loader = utils.data.DataLoader(imbalanced_train_dataset, shuffle=True, batch_size=64)
val_loader = utils.data.DataLoader(val_subset, shuffle=False, batch_size=64)


class_distribution_train = calculate_class_distribution(train_loader)
print("Training Dataset Class Distribution:", class_distribution_train)

class_distribution_val = calculate_class_distribution(val_loader)
print("Validation Dataset Class Distribution:", class_distribution_val)


test_dataset = datasets.CIFAR10("~/.torch/data", train=False, transform=myCNN.get_input_transform(mu, cov))
test_loader = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)

cnn_model = myCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
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
