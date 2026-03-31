#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly.
This is a simplified version that tests:
1. Data loading
2. Model forward pass
3. Loss computation
4. Backward pass
5. AUC computation
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def build_baseline_cnn(num_classes: int):
    import torchvision.models as models
    model = models.densenet121(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def test_pipeline():
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
    
    # Use smaller subsets for testing
    train_files = train_files[:100]  # Use only 100 training samples
    val_files = val_files[:50]      # Use only 50 validation samples
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=train_transform, sample_filter=train_files, validate=True)
    val_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=val_transform, sample_filter=val_files, validate=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test data loading
    print("\nTesting data loading...")
    sample_img, sample_label, sample_name = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Number of diseases in sample: {sample_label.sum()}")
    
    # Build model
    print("\nBuilding model...")
    model = build_baseline_cnn(num_classes=14).to(device)
    
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Trainable params: {count_trainable_params(model)}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        test_images, test_labels, _ = next(iter(train_loader))
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = model(test_images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    
    # Set up training
    print("\nSetting up training...")
    train_labels = np.array([train_dataset.label_dict[img] for img in train_dataset.samples])
    pos_weights = compute_pos_weights(torch.tensor(train_labels, dtype=torch.float32))
    print(f"Positive weights: {pos_weights}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    images, labels, _ = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    loss = criterion(logits, labels)
    print(f"Initial loss: {loss.item():.4f}")
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Test second training step to verify loss decreases
    print("\nTesting second training step...")
    optimizer.zero_grad()
    outputs = model(images)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    loss = criterion(logits, labels)
    print(f"Second loss: {loss.item():.4f}")
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Test evaluation
    print("\nTesting evaluation...")
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    print(f"Evaluation labels shape: {all_labels.shape}")
    print(f"Evaluation preds shape: {all_preds.shape}")
    
    # Test AUC computation
    print("\nTesting AUC computation...")
    try:
        auc_scores = roc_auc_score(all_labels, all_preds, average=None)
        mean_auc = auc_scores.mean()
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
        
        print(f"Mean AUC: {mean_auc:.4f}")
        print(f"Macro AUC: {macro_auc:.4f}")
        print(f"Per-class AUC: {auc_scores}")
        print("\n✓ All tests passed!")
        return True
    except Exception as e:
        print(f"✗ AUC computation failed: {e}")
        return False

def compute_pos_weights(labels: torch.Tensor) -> torch.Tensor:
    num_positives = labels.sum(dim=0)
    num_negatives = len(labels) - num_positives
    return num_negatives / (num_positives + 1e-6)

if __name__ == "__main__":
    test_pipeline()