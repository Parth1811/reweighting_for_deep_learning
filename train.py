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
#from model import myCNN
from alexNet import AlexNet
from utils import (create_folders, compute_accuracy, get_run_uuid, stratified_split, setup_logger, calculate_class_distribution)


RUN_UUID = get_run_uuid()

parser = argparse.ArgumentParser("Learning Reweight for Classification - Toy Problem")
parser.add_argument("-n", "--name", type=str, default=RUN_UUID, help="Name of the run")
parser.add_argument("-e", "--epoch", type=int, default=50, help="Number of Epoch")
parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="Learning Rate")
parser.add_argument("-s", "--save", action="store_true", default=True, help="Save the model after running")
parser.add_argument("-l", "--load", type=str, default=None, help="Load the model from path before running")
parser.add_argument("-ld", "--log_dir", type=str, default='./run_data/log', help='directory of log')
parser.add_argument("-sp", "--save-path", type=str, default="./run_data", help="Path to save the model and checkpoints")
parser.add_argument("-c", "--checkpoint", type=int, default=10, help="Set checkpoint interval")
parser.add_argument("-sc", "--save-checkpoints", action="store_true", default=False, help="Save the model checkpoints while running")
parser.add_argument("-ca", "--print-checkpoint-accuracy", action="store_true", default=False, help="Print model accuracy for checkpoints")
args = parser.parse_args()

create_folders([args.save_path, args.log_dir])

# CIFAR-10 mean and cov values
mu = (0.5, 0.5, 0.5)
cov = (0.5, 0.5, 0.5)

# ----------------- Load the CIFAR-10 Train Dataset ---------------------------
train_dataset = datasets.CIFAR10("~/.torch/data", train=True, transform=AlexNet.get_input_transform(mu, cov))
train_idx, val_idx = stratified_split(train_dataset, test_size=0.001)
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)

imbalanced_train_dataset = ImbalancedCIFAR10Dataset(train_subset, dominant_class=2, percentage=0.9)
train_loader = utils.data.DataLoader(imbalanced_train_dataset, shuffle=True, batch_size=64)
val_loader = utils.data.DataLoader(val_subset, shuffle=False, batch_size=64)

# ----------------- Load the CIFAR-10 Test Dataset ---------------------------
test_dataset = datasets.CIFAR10("~/.torch/data", train=False, transform=AlexNet.get_input_transform(mu, cov))
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


cnn_model = AlexNet()
if args.load is not None:
    cnn_model.load_state_dict(torch.load(args.load))

optimizer = optim.SGD(cnn_model.parameters(), lr=args.learning_rate, momentum=0.9)
loss_fn_mean_red = nn.CrossEntropyLoss(reduction="mean")
loss_fn_no_red = nn.CrossEntropyLoss(reduction="none")


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