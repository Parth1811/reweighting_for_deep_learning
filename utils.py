from datetime import datetime
import torch
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