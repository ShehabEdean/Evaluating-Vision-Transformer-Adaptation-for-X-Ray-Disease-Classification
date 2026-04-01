#!/usr/bin/env python3
"""
Training script optimized for Google Colab with GPU support.
This script assumes:
1. You've mounted Google Drive
2. Your dataset is in /content/drive/MyDrive/Xray/
3. You've cloned your repo to Colab
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
    except Exception as e:
        print(f"AUC computation error: {e}")
        return None, None, None, val_loss

def main():
    set_seed(RANDOM_SEED)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Warn if using CPU
    if device.type == 'cpu':
        print("⚠️  WARNING: Running on CPU - training will be very slow!")
        print("Go to Runtime → Change runtime type → GPU for 10-50x speedup")
    
    # LOCAL COLAB PATHS - using repository's own dataset structure
    CSV_PATH = "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/dataset/labels.csv"
    IMAGE_DIRS = [
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_001/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_002/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_003/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_004/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_005/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_006/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_007/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_008/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_009/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_010/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_011/images",
        "/content/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/data/images_012/images",
    ]
    
    print("Loading and splitting data...")
    train_files, val_files = split_by_patient(CSV_PATH, train_ratio=0.8, seed=RANDOM_SEED)
    
    # For quick testing in Colab, use smaller subsets
    # Remove these lines for full training
    print("Using reduced dataset for testing...")
    train_files = train_files[:1000]   # 1000 training samples (increased from 500)
    val_files = val_files[:300]      # 300 validation samples (increased from 100)
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=train_transform, sample_filter=train_files, validate=True)
    val_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=val_transform, sample_filter=val_files, validate=False)
    
    # Reduced workers to avoid warnings and improve stability
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_baseline_cnn(num_classes=14).to(device)
    
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Trainable params: {count_trainable_params(model)}")
    
    # Set up training
    train_labels = np.array([train_dataset.label_dict[img] for img in train_dataset.samples])
    pos_weights = compute_pos_weights(torch.tensor(train_labels, dtype=torch.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    # Train for 5 epochs (better results while still quick)
    num_epochs = 5
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.2f}s")
        
        auc_scores, mean_auc, macro_auc, val_loss = evaluate(model, val_loader, device, criterion)
        if mean_auc is not None:
            print(f"Mean AUC: {mean_auc:.4f}, Macro AUC: {macro_auc:.4f}")
            print(f"Per-class AUC: {auc_scores}")
            
            # Update scheduler
            scheduler.step(mean_auc)
            
            # Save best model
            if mean_auc > best_auc:
                best_auc = mean_auc
                print("New best model!")
        print()
    
    print(f"Training completed! Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()