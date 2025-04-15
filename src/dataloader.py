import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


class MyDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        assert split in ["train", "test", "valid"], \
            f"{split} is not a valid split"
        self.split = split
        self.transform = transform

        self.imagedir = os.path.join(self.root_dir, self.split)

        self.annotations = []
        self.img_to_annotations = {}
        self.images_info = []

        if split == "test":
            self.image_filenames = sorted(
                [f for f in os.listdir(self.imagedir) if f.endswith('.png')],
                key=lambda x: int(x.split('.')[0]))
            return
        annotation_file = os.path.join(root_dir, f'{split}.json')

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.categories = {cat['id']: int(cat['name'])
                           for cat in data['categories']}

        self.images_info = {img['id']: img for img in data['images']}

        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_annotations.keys():
                self.img_to_annotations[img_id] = []
            self.img_to_annotations[img_id].append(ann)

        self.image_ids = list(self.images_info.keys())

    def __len__(self):
        if self.split != 'test':
            return len(self.image_ids)
        else:
            return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.split != 'test':
            img_id = self.image_ids[idx]
            img_info = self.images_info[img_id]
            img_path = os.path.join(self.imagedir, img_info['file_name'])

            # Load annotations for this image
            annotations = self.img_to_annotations.get(img_id, [])

            # Extract bounding boxes and categories
            boxes = []
            categories = []

            for ann in annotations:
                # COCO format bbox is [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format for training
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])

                # Get digit class (0-9)
                category_id = ann['category_id']
                # digit_class = self.categories[category_id]
                # categories.append(digit_class)
                categories.append(category_id)

            # If no annotations,
            # create empty tensors
            if not boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                categories = torch.zeros(0, dtype=torch.int64)
                digit_count = 0
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                categories = torch.tensor(categories, dtype=torch.int64)

                # Get the digit count (task 2) -
                # extracting the number represented by the digits
                # Sort boxes from left to right
                if len(boxes) > 0:
                    sorted_indices = torch.argsort(boxes[:, 0])
                    categories = categories[sorted_indices]
                    boxes = boxes[sorted_indices]

                # Construct the number from the digits
                digits = [str(cat.item()) for cat in categories]
                digit_count = int(''.join(digits)) if digits else 0

        else:
            # For test set, we only load the image
            img_path = os.path.join(self.imagedir, self.image_filenames[idx])
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Placeholder
            categories = torch.zeros(0, dtype=torch.int64)    # Placeholder
            digit_count = 0  # Placeholder

        # Load image
        image = Image.open(img_path).convert('RGB')

        original_width, original_height = image.width, image.height
        new_width, new_height = 128, 64
        scale_width = new_width / image.width
        scale_height = new_height / image.height
        image = F.resize(image, [new_height, new_width])

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_width  # Scale x coordinates
            boxes[:, [1, 3]] *= scale_height  # Scale y coordinates

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Create a sample dictionary
        sample = {
            'image': image,
            'boxes': boxes,
            'categories': categories,
            'digit_count': digit_count,
            'image_id': img_id if self.split != 'test'
            else int(self.image_filenames[idx].split('.')[0]),
            'original_width': original_width,
            'original_height': original_height
        }

        return sample


def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    categories = [item['categories'] for item in batch]
    digit_counts = [item['digit_count'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    original_widths = [item['original_width'] for item in batch]
    original_heights = [item['original_height'] for item in batch]

    # Stack images into a batch
    images = torch.stack(images, 0)

    # Return as a dictionary
    return {
        'images': images,
        'boxes': boxes,
        'categories': categories,
        'digit_counts': torch.tensor(digit_counts),
        'image_ids': torch.tensor(image_ids),
        'original_widths': torch.tensor(original_widths),
        'original_heights': torch.tensor(original_heights)
    }
