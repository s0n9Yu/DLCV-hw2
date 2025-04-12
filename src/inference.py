import os
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from torch.optim import SGD, Adam
from tqdm import tqdm
import argparse

from dataloader import MyDataset, collate_fn
from model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="model path")
args = parser.parse_args()

train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_dataset = MyDataset("data", split="test", transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = "cuda"
model = torch.load(args.model).to(device)

model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        images = [image.to(device) for image in batch["images"]]
        image_ids = batch["image_ids"]

        prediction = model(images)
        for i, pred in enumerate(prediction):
            pred_dict = {
                'image_id': image_ids[i].item(),
                'boxes': pred['boxes'].cpu().numpy(),
                'scores': pred['scores'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy(),
                'original_width': batch["original_widths"], 
                'original_height': batch["original_heights"]
            }
            predictions.append(pred_dict)

# task 1
result = []
    
for pred in predictions:
    image_id = pred['image_id']
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']
    original_width = pred['original_width']
    original_height = pred['original_height']
    
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Confidence threshold
            # Convert [x1, y1, x2, y2] to [x_min, y_min, width, height]
            x1, y1, x2, y2 = box
            x1 *= original_width / 128
            x2 *= original_width / 128
            y1 *= original_height / 64
            y2 *= original_height / 64

            width = x2 - x1
            height = y2 - y1
            
            result.append({
                'image_id': int(image_id),
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(score),
                'category_id': int(label)
            })

with open("pred.json", 'w') as f:
    json.dump(result, f)


# task 2
# Filter by confidence
def recognize_number(boxes, labels, scores, threshold=0.6):
    """
    Recognize the entire number from individual digit detections
    """
    # Filter by confidence
    valid_indices = scores > threshold
    if not valid_indices.any():
        return -1
    
    
    filtered_boxes = boxes[valid_indices]
    filtered_labels = labels[valid_indices]
    filtered_scores = scores[valid_indices]

    # Step 2: Select top-k highest confidence detections
    topk_indices = filtered_scores.argsort()[-3:]
    topk_boxes = filtered_boxes[topk_indices]
    topk_labels = filtered_labels[topk_indices]

    filtered_boxes = topk_boxes[topk_indices]
    filtered_labels = topk_labels[topk_indices]

    
    # Sort boxes from left to right
    sorted_indices = filtered_boxes[:, 0].argsort()
    sorted_labels = filtered_labels[sorted_indices]
    
    # Convert labels to digits (subtract 1 because category_id starts from 1)
    digits = [str(int(l) - 1) for l in sorted_labels]

    
    if not digits:
        return -1
    
    try:
        number = int(''.join(digits))
        return number
    except ValueError:
        return -1

results = {}

for pred in predictions:
    image_id = pred['image_id']
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']
    
    number = recognize_number(boxes, labels, scores)
    results[image_id] = number

# Write to CSV file
with open("pred.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'pred_label'])
    for image_id, number in sorted(results.items()):
        writer.writerow([image_id, number])