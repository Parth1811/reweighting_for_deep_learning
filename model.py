import torch
from torch import nn
from torchvision import transforms

first_layer_size = 32
second_layer_size = 64
third_layer_size = 64
dense_layer_size = 512
OUTPUT_CLASSES = 10

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(3, first_layer_size, kernel_size=3, padding=1)
        self.conv_layer_2 = nn.Conv2d(first_layer_size, second_layer_size, kernel_size=3, padding=1)
        self.conv_layer_3 = nn.Conv2d(second_layer_size, third_layer_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(third_layer_size * 4 * 4, dense_layer_size)
        self.fc2 = nn.Linear(dense_layer_size, OUTPUT_CLASSES)

    def forward(self, x):
        x1 = self.pool(torch.relu(self.conv_layer_1(x)))
        x2 = self.pool(torch.relu(self.conv_layer_2(x1)))
        x3 = self.pool(torch.relu(self.conv_layer_3(x2)))
        x4 = x3.view(-1, third_layer_size * 4 * 4)
        x5 = torch.relu(self.fc1(x4))
        return self.fc2(x5)
    
    @staticmethod
    def get_input_transform(mean, cov):
        return transforms.Compose([
            transforms.Resize((first_layer_size, first_layer_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, cov)
        ])
