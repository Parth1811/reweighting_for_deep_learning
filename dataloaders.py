
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset 
import numpy as np

class ImbalancedCIFAR10Dataset(Dataset):
    def __init__(self, dataset, dominant_class, percentage = 0.9):
        self.dataset = dataset
        self.dominant_class = dominant_class
        self.percentage_other_classes = 1 - percentage
        self.dominant_indices = []
        self.indices_other_classes = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label == 2:
                self.dominant_indices.append(i)
            else:
                self.indices_other_classes.append(i)
        
        total_dom = len(self.dominant_indices)
        total_other_classes = int(total_dom  / percentage - total_dom) / 9

        np.random.shuffle(self.indices_other_classes)
        self.indices_other_classes = self.indices_other_classes[:int(total_other_classes * 9)]

        self.indices = self.dominant_indices + self.indices_other_classes
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.dataset[original_index]

    def __len__(self):
        return len(self.indices)