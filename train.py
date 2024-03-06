import torch
from torch import nn, optim, utils
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import random
import higher 
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


cnn_model = myCNN()#.to(device)
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# Assume train_loader, val_loader are defined and loaded with your datasets

for epoch in range(60):
    cnn_model.train()

    for img, labels in train_loader:
        #img, labels = img.to(device), labels.to(device)

        # Initialize eps for each example in the batch to be learnable
        eps = torch.zeros(img.size(0), requires_grad=True, device=device)
        
        optimizer.zero_grad()

        with higher.innerloop_ctx(cnn_model, optimizer) as (meta_model, meta_opt):
            # Forward pass on the meta-model with training data
            meta_outputs = meta_model(img)
            loss_fn.reduction = 'none'
            meta_loss = loss_fn(meta_outputs, labels)

            # Use eps to modulate loss
            weighted_meta_loss = torch.sum(eps * meta_loss)

            # Step through meta-optimizer
            meta_opt.step(weighted_meta_loss)

            # Now compute validation loss
            val_img, val_labels = next(iter(val_loader))  # Assuming sufficient validation data
            val_img, val_labels = val_img, val_labels
            val_outputs = meta_model(val_img)
            loss_fn.reduction = 'mean'
            val_loss = loss_fn(val_outputs, val_labels)

            # Compute gradients of eps based on validation loss
            eps_grads = torch.autograd.grad(val_loss, eps)[0].detach()
        
        # Use eps_grads to adjust training procedure
        w_tilde = torch.clamp(-eps_grads, min=0)
        l1_norm = torch.sum(w_tilde)
        w = w_tilde / l1_norm if l1_norm != 0 else w_tilde

        outputs = cnn_model(img)
        loss_fn.reduction = 'none'
        loss = loss_fn(outputs, labels)

        # Apply computed weights to loss
        weighted_loss = torch.sum(w * loss)
        weighted_loss.backward()
        optimizer.step()

    print(f"Running epoch: {epoch} | Loss: {weighted_loss.item()}")
    if epoch % 5 == 0 and epoch != 0:
        compute_accuracy()

compute_accuracy()