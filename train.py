import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import random

from loss import get_weighted_loss
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
class_weights, weighted_loss_fn = get_weighted_loss(None)
unweighted_loss_fn = nn.CrossEntropyLoss()

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


    # TODO(@Parth): This only in case of CIFAR-10 where validation set only has one mini batch.
    #               Need to figure out a way to get random samples from the val_loader
    val_img, val_labels = next(iter(val_loader))

    for img, labels in train_loader:

        optimizer.zero_grad()
        out = cnn_model(img)
        loss = weighted_loss_fn(out, labels)
        current_params = [param.clone().detach() for param in cnn_model.parameters()]
        loss.backward()
        optimizer.step()
        grads_a = [param.grad for param in cnn_model.parameters()]



        optimizer.zero_grad()
        val_out = cnn_model(val_img)
        loss = unweighted_loss_fn(val_out, val_labels)
        loss.backward()
        optimizer.step()
        grads_b = [param.grad for param in cnn_model.parameters()]

        grads_ex_wts = torch.autograd.grad(grads_a, [class_weights], grads_b, create_graph=True)


        weights = torch.max(-grads_ex_wts, 0)
        weights_norm = torch.sum(weights)
        weights_norm = weights_norm if weights_norm != 0 else 0
        class_weights = weights / weights_norm


        class_weights, weighted_loss_fn = get_weighted_loss(class_weights)
        
        with torch.no_grad():
            for param, current_param in zip(cnn_model.parameters(), current_params):
                param.copy_(current_param)

        optimizer.zero_grad()
        out = cnn_model(img)
        loss = weighted_loss_fn(out, labels)
        current_params = [param.clone().detach() for param in cnn_model.parameters()]
        loss.backward()
        optimizer.step()

    print(f"Running epoch: {epoch} | Loss: {loss.item()}")

    if epoch % 5 == 0 and epoch != 0:
        compute_accuracy()

compute_accuracy()