import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from torch.optim import SGD, Adam
from tqdm import tqdm

from dataloader import MyDataset, collate_fn
from model import MyModel

import wandb
if os.environ.get('WANDB_API_KEY') is not None:
    useWandb = True
    wandb.login()
else:
    useWandb = False


train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = MyDataset("data", split="train", transform=train_transform)
valid_dataset = MyDataset("data", split="valid", transform=train_transform)
test_dataset = MyDataset("data", split="test", transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

epochs = 5
lr = 0.001
device = "cuda"
model = MyModel().to(device)
optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=0.0005
            )

if useWandb:
    run = wandb.init(
        # Set the project where this run will be logged
        project="dlcv-hw2",
        # Track hyperparameters and run metadata
        config={
            "batch size": 4,
            "learning_rate": lr,
            "epochs": epochs,
            "model": str(model),
        },
        notes="no normalization",
    )
    model_name = wandb.run.name
else:
    model_name = "model"

os.makedirs("checkpoint", exist_ok=True)

for epoch in tqdm(range(epochs), desc = "epochs"):
    

    model.train()
    train_loss = 0
    valid_loss = 0
    for batch in tqdm(train_loader, desc = "train"):
        images = list(image.to(device) for image in batch['images'])
        
        targets = model.convert_targets(
            batch['boxes'], 
            batch['categories']
        )
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #print("loss: ", losses_reduced)
        train_loss += losses_reduced
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    # validation
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc = "valid"):
            images = list(image.to(device) for image in batch['images'])
            
            targets = model.convert_targets(
                batch['boxes'], 
                batch['categories']
            )
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            valid_loss += losses_reduced
        valid_loss /= len(valid_loader)
    torch.save(model, f"checkpoint/{model_name}_epoch{epoch}.ckpt")
    if useWandb:
        wandb.log({"train loss": train_loss,
                   "valid loss": valid_loss})
        
        
torch.save(model, f"checkpoint/{model_name}.ckpt")
print(f"Save the model to checkpoint/{model_name}.ckpt")