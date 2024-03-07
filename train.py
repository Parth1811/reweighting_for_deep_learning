import torch
from torch import nn, optim, utils
from torchvision import datasets
from torch.utils.data import Subset
import higher 

from dataloaders import ImbalancedCIFAR10Dataset
from loss import get_weighted_loss
from model import myCNN
from utils import calculate_class_distribution, compute_accuracy, stratified_split


# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

# ----------------- Load the CIFAR-10 Train Dataset ---------------------------
train_dataset = datasets.CIFAR10("~/.torch/data", train=True, transform=myCNN.get_input_transform(mu, cov))
train_idx, val_idx = stratified_split(train_dataset, test_size=0.001)
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)

imbalanced_train_dataset = ImbalancedCIFAR10Dataset(train_subset, dominant_class=2, percentage=0.9)
train_loader = utils.data.DataLoader(imbalanced_train_dataset, shuffle=True, batch_size=64)
val_loader = utils.data.DataLoader(val_subset, shuffle=False, batch_size=64)

# ----------------- Check the CLass Distributions ---------------------------
class_distribution_train = calculate_class_distribution(train_loader)
print("Training Dataset Class Distribution:", class_distribution_train)

class_distribution_val = calculate_class_distribution(val_loader)
print("Validation Dataset Class Distribution:", class_distribution_val)


# ----------------- Load the CIFAR-10 Test Dataset ---------------------------
test_dataset = datasets.CIFAR10("~/.torch/data", train=False, transform=myCNN.get_input_transform(mu, cov))
test_loader = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)

cnn_model = myCNN()
class_weights, weighted_loss_fn = get_weighted_loss(None)
unweighted_loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cpu")


cnn_model = myCNN()#.to(device)
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
loss_fn_mean_red = nn.CrossEntropyLoss(reduction="mean")
loss_fn_no_red = nn.CrossEntropyLoss(reduction="none")

for epoch in range(60):
    cnn_model.train()

    for img, labels in train_loader:
        batch_size = img.shape[0]
        epsilon = torch.zeros(batch_size, requires_grad=True, device=device)        
        optimizer.zero_grad()

        # Reference for using pytorch higher: github.com/TinfoilHat0/Learning-to-Reweight-Examples-for-Robust-Deep-Learning-with-PyTorch-Higher/
        with higher.innerloop_ctx(cnn_model, optimizer) as (meta_model, meta_opt):
            
            # Update the meta model with intermidiate theta
            meta_outputs = meta_model(img)
            meta_loss = loss_fn_no_red(meta_outputs, labels)
            weighted_meta_loss = torch.sum(epsilon * meta_loss)
            meta_opt.step(weighted_meta_loss)

            # Compute Valdiation loss using the intermidiate model
            # TODO(Parth): next(iter(val_loader)) will always give the first element of val_loader
            #              This only works in case of CIFAR-10 because we just have one mini-batch in val_loader
            val_img, val_labels = next(iter(val_loader))
            val_outputs = meta_model(val_img)
            val_loss = loss_fn_mean_red(val_outputs, val_labels)

            # Compute gradients validation loss wrt epsilon
            eps_grads = torch.autograd.grad(val_loss, epsilon)[0].detach()
        
        # Use eps_grads to adjust training procedure
        weights = torch.clamp(-eps_grads, min=0)
        weights_norm = torch.sum(weights)
        weights_norm = weights_norm if weights_norm != 0 else 1
        weights = weights / weights_norm

        outputs = cnn_model(img)
        
        loss = loss_fn_no_red(outputs, labels)
        weighted_loss = torch.sum(weights * loss)
        weighted_loss.backward()
        
        optimizer.step()

    print(f"Running epoch: {epoch} | Loss: {weighted_loss.item()}")
    if epoch % 5 == 0 and epoch != 0:
        compute_accuracy(cnn_model, test_loader)

compute_accuracy(cnn_model, test_loader)