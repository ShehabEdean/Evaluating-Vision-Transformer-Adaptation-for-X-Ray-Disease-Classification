#!/usr/bin/env python3
"""
Flexible training script that helps find your dataset in Google Drive.
"""

import os
import subprocess
import sys

def find_dataset_in_drive():
    """Search for dataset files in Google Drive"""
    print("🔍 Searching for dataset in Google Drive...")
    
    # Check if drive is mounted
    if not os.path.exists('/content/drive'):
        print("❌ Google Drive not mounted!")
        print("Run: from google.colab import drive; drive.mount('/content/drive')")
        return None, None
    
    # Search for labels.csv
    try:
        result = subprocess.run(['find', '/content/drive', '-name', 'labels.csv'], 
                              capture_output=True, text=True, timeout=30)
        csv_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if not csv_files:
            print("❌ No labels.csv found in Google Drive")
            return None, None
        
        print(f"✅ Found {len(csv_files)} labels.csv files:")
        for i, file in enumerate(csv_files):
            print(f"  {i+1}. {file}")
        
        # Use the first one found
        csv_path = csv_files[0]
        print(f"\n📋 Using CSV: {csv_path}")
        
        # Find image directories near the CSV file
        dataset_dir = os.path.dirname(csv_path)
        base_dir = os.path.dirname(dataset_dir)
        
        # Look for images_001 to images_012 directories
        image_dirs = []
        for i in range(1, 13):
            img_dir = os.path.join(base_dir, f"images_{i:03d}", "images")
            if os.path.exists(img_dir):
                image_dirs.append(img_dir)
        
        if not image_dirs:
            print("❌ No image directories found")
            # Try alternative pattern
            for i in range(1, 13):
                img_dir = os.path.join(base_dir, f"data/images_{i:03d}/images")
                if os.path.exists(img_dir):
                    image_dirs.append(img_dir)
        
        if image_dirs:
            print(f"✅ Found {len(image_dirs)} image directories:")
            for dir in image_dirs[:3]:  # Show first 3
                print(f"  - {dir}")
            if len(image_dirs) > 3:
                print(f"  ... and {len(image_dirs)-3} more")
        else:
            print("❌ No image directories found")
            return csv_path, None
            
        return csv_path, image_dirs
        
    except subprocess.TimeoutExpired:
        print("⏱️ Search timed out - Google Drive might be large")
        return None, None
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return None, None

def main():
    csv_path, image_dirs = find_dataset_in_drive()
    
    if not csv_path or not image_dirs:
        print("\n🛠️ Manual Setup Required:")
        print("1. Check your Google Drive structure")
        print("2. Update the paths in colab_train.py manually")
        print("3. Common paths:")
        print("   - /content/drive/MyDrive/Xray/dataset/labels.csv")
        print("   - /content/drive/MyDrive/data/Xray/labels.csv")
        print("   - /content/drive/MyDrive/Colab Notebooks/Xray/labels.csv")
        return
    
    print(f"\n🚀 Ready to train with:")
    print(f"   CSV: {csv_path}")
    print(f"   Images: {len(image_dirs)} directories")
    
    # Now run the actual training
    print("\n🔄 Starting training...")
    
    # Import training modules
    import random
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    
    sys.path.insert(0, 'src')
    from dataset import ChestXRayDataset, split_by_patient
    from transforms import get_train_transform, get_val_transform
    
    RANDOM_SEED = 42
    
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading and splitting data...")
    train_files, val_files = split_by_patient(csv_path, train_ratio=0.8, seed=RANDOM_SEED)
    
    # Use small subset for testing
    train_files = train_files[:50]
    val_files = val_files[:20]
    
    train_dataset = ChestXRayDataset(csv_path, image_dirs, transform=get_train_transform(), 
                                    sample_filter=train_files, validate=True)
    val_dataset = ChestXRayDataset(csv_path, image_dirs, transform=get_val_transform(),
                                  sample_filter=val_files, validate=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Simple training setup
    from torchvision.models import densenet121
    
    model = densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train for 2 epochs
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/2 - Loss: {running_loss/len(train_loader):.4f}")
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()