import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2

from dataloader import MyDataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = MyDataset("data", split="train", transform=train_transform)
tmp = dataset[200]
print(tmp["image"].shape)
print(tmp["boxes"].shape)

new_img_PIL = transforms.ToPILImage()(tmp["image"]).convert('RGB')
new_img_PIL = np.array(new_img_PIL)

# Draw red boxes
for box in tmp["boxes"]:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(new_img_PIL, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

print(new_img_PIL.shape)
cv2.imwrite("img.jpg", new_img_PIL)