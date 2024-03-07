import copy
from datetime import datetime
import logging
import torch
from tqdm import tqdm
import sys
import os
from sklearn.model_selection import train_test_split

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

def compute_accuracy(cnn_model, test_loader):
    cnn_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total * 100:.2f}%")

def get_run_uuid():
    return datetime.now().strftime("%d_%m_%Y__%H_%M_%S")


def create_folders(folder_list):
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)


class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno
        if(levelno >= 50):  # CRITICAL / FATAL
            color = '\x1b[31m'  # red
        elif(levelno >= 40):  # ERROR
            color = '\x1b[31m'  # red
        elif(levelno >= 30):  # WARNING
            color = '\x1b[33m'  # yellow
        elif(levelno >= 20):  # INFO
            color = '\x1b[36m'  # cyan
        elif(levelno >= 15):  # DEBUG
            color = '\x1b[35m'  # pink
        elif(levelno == 10):  # notification
            color = '\x1b[32m'  # green
        else:  # NOTSET and anything else
            color = '\x1b[0m'  # normal
        myrecord.msg = color + str(myrecord.msg) + '\x1b[0m'  # normal
        logging.StreamHandler.emit(self, myrecord)


def setup_logger(log_file_path):
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(message)s',
        datefmt='%m/%d/%y %I:%M:%S %p'
    )

    file_handler = logging.FileHandler(
        log_file_path,
        mode='w'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)
    out_stream = type("TqdmStream", (), {'file': sys.stdout, 'write':write})()

    stdout_handler = ColoredConsoleHandler(out_stream)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    logging.addLevelName(15, "Inside Iteration")     # pink
    logging.addLevelName(35, "Epoch End")            # yellow
    logging.addLevelName(45, "Model Save")           # red
    logging.addLevelName(25, "Log saved")            # cyan
    logging.addLevelName(12, "Notification")            # cyan

    return logger