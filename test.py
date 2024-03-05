import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from PIL import Image
import urllib.request
import json
import os

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

class_labels = {}
with open("imagenet_labels.json", 'r') as f:
    class_labels = json.load(f)


# Using mean and std from the imageNet dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = 'image.jpg'
if not os.path.exists(image_path):
    print("Downloading a cat image from the web........")
    image_url = 'https://i.redd.it/drawing-the-cat-as-random-cat-memes-day-57-original-in-v0-ww64vba9zb6a1.jpg?width=1280&format=pjpg&auto=webp&s=bb1ed22c02f1d6a514e1da3fc3a9b244f574db8d'
    urllib.request.urlretrieve(image_url, image_path)
image = Image.open(image_path)
image = transform(image).unsqueeze(0)       # unsqueeze is used for batch processing


# Pass image through vgg16
with torch.no_grad():
    output = vgg16(image)


probabilities = torch.nn.functional.softmax(output[0], dim=0)
_, predicted_class = torch.max(probabilities, 0)
predicted_class_name = class_labels[str(predicted_class.item())]

print(f"Predicted class: {predicted_class.item()} - '{predicted_class_name}', Confidence: {probabilities[predicted_class].item()}")