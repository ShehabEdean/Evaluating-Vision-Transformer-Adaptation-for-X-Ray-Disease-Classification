#!/usr/bin/env python3
"""
Quick test to verify the training pipeline works with minimal data.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, 'src')

from dataset import ChestXRayDataset, DISEASE_LABELS, split_by_patient
from transforms import get_train_transform, get_val_transform

RANDOM_SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_baseline_cnn(num_classes: int):
    import torchvision.models as models
    model = models.densenet121(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def compute_pos_weights(labels: torch.Tensor) -> torch.Tensor:
    num_positives = labels.sum(dim=0)
    num_negatives = len(labels) - num_positives
    return num_negatives / (num_positives + 1e-6)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            if criterion:
                loss = criterion(logits, labels)
                running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    val_loss = running_loss / len(dataloader) if criterion else None
    
    try:
        auc_scores = roc_auc_score(all_labels, all_preds, average=None)
        mean_auc = auc_scores.mean()
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
        return auc_scores, mean_auc, macro_auc, val_loss
    except:
        return None, None, None, val_loss

def main():
    set_seed(RANDOM_SEED)
    device = torch.device('cpu')
    
    CSV_PATH = "dataset/labels.csv"
    IMAGE_DIRS = [
        "data/images_001/images",
        "data/images_002/images",
        "data/images_003/images",
        "data/images_004/images",
        "data/images_005/images",
        "data/images_006/images",
        "data/images_007/images",
        "data/images_008/images",
        "data/images_009/images",
        "data/images_010/images",
        "data/images_011/images",
        "data/images_012/images",
    ]
    
    print("Loading and splitting data...")
    train_files, val_files = split_by_patient(CSV_PATH, train_ratio=0.8, seed=RANDOM_SEED)
    
    # Use very small subsets for quick testing
    train_files = train_files[:50]   # 50 training samples
    val_files = val_files[:20]      # 20 validation samples
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=train_transform, sample_filter=train_files, validate=True)
    val_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=val_transform, sample_filter=val_files, validate=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_baseline_cnn(num_classes=14).to(device)
    
    # Set up training
    train_labels = np.array([train_dataset.label_dict[img] for img in train_dataset.samples])
    pos_weights = compute_pos_weights(torch.tensor(train_labels, dtype=torch.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train for 2 epochs
    num_epochs = 2
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.2f}s")
        
        auc_scores, mean_auc, macro_auc, val_loss = evaluate(model, val_loader, device, criterion)
        if mean_auc is not None:
            print(f"Mean AUC: {mean_auc:.4f}, Macro AUC: {macro_auc:.4f}")
            print(f"Per-class AUC: {auc_scores}")
        print()
    
    print("✓ Training test completed successfully!")

if __name__ == "__main__":
    main()