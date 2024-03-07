import argparse
import torch
from torch import nn, optim, utils
from torchvision import datasets
from torch.utils.data import Subset
import higher
import os
from tqdm import tqdm

from dataloaders import ImbalancedCIFAR10Dataset
from loss import get_weighted_loss
from model import myCNN

# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

train_dataset = datasets.CIFAR10("~/.torch/data", train=True, transform=myCNN.get_input_transform(mu, cov))
train_loader = utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)

test_dataset = datasets.CIFAR10("~/.torch/data", train=False, transform=myCNN.get_input_transform(mu, cov))
test_loader = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)


# Setting up the logging modules
# Log levels
# Inside Iteration 15
# Epoch End 35
# Model save 45
# Log Save 25
progressbar = tqdm(range(args.epoch), desc=f"Training Toy Problem Run - {args.name}", ascii=False, dynamic_ncols=True, colour="yellow")
log_file_path = os.path.join(args.log_dir, f"run_{args.name}.log")
logger = setup_logger(log_file_path)
logger.info(args)


# ----------------- Check the CLass Distributions ---------------------------
class_distribution_train = calculate_class_distribution(train_loader)
logger.info(f"Training Dataset Class Distribution: {class_distribution_train}")

class_distribution_val = calculate_class_distribution(val_loader)
logger.info(f"Validation Dataset Class Distribution: {class_distribution_val}")


cnn_model = myCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cpu")
logger.info('Training has Started')
for epoch in range(args.epoch):
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

    logger.log(35, f"Epoch: {epoch} | Loss: {weighted_loss.item()}")
    progressbar.update(1)
    if epoch % args.checkpoint == 0 and epoch != 0:
        if args.print_checkpoint_accuracy:
            acc = compute_accuracy(cnn_model, test_loader)
            logger.info(f"Accuracy: {acc:.2f}%")
        if args.save_checkpoints:
            save_path = os.path.join(args.save_path, f"{args.name}_checkpoint_{epoch // args.checkpoint}.pth")
            logger.log(45, f"Saving model checkpoint at {save_path}")
            torch.save(cnn_model.state_dict(), save_path)

acc = compute_accuracy(cnn_model, test_loader)
logger.info(f"Accuracy: {acc:.2f}%")
if args.save:
    save_path = os.path.join(args.save_path, f"{args.name}.pth")
    logger.log(45, f"Saving model at {save_path}")
    torch.save(cnn_model.state_dict(), os.path.join(args.save_path, f"{args.name}.pth"))

logger.info('Training is finished')
logger.log(25, "Saved log file " + log_file_path)
progressbar.close()