import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

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


def compute_pos_weights(labels: torch.Tensor) -> torch.Tensor:
    num_positives = labels.sum(dim=0)
    num_negatives = len(labels) - num_positives
    return num_negatives / (num_positives + 1e-6)


def build_baseline_cnn(num_classes: int):
    import torchvision.models as models
    model = models.densenet121(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def build_vit_model(strategy: str = "full", num_classes: int = 14):
    from transformers import ViTForImageClassification
    from peft import LoraConfig, get_peft_model
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    if strategy == "full":
        pass 

    elif strategy == "partial":
        for param in model.vit.embeddings.parameters():
            param.requires_grad = False
        for layer in model.vit.encoder.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False

    elif strategy == "peft_lora":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_labels = torch.cat([l for l in all_labels]).numpy() if isinstance(all_labels[0], torch.Tensor) else torch.cat([torch.tensor(l) for l in all_labels]).numpy()
    all_preds = torch.cat([p for p in all_preds]).numpy() if isinstance(all_preds[0], torch.Tensor) else torch.cat([torch.tensor(p) for p in all_preds]).numpy()

    try:
        auc_scores = roc_auc_score(all_labels, all_preds, average=None)
        mean_auc = auc_scores.mean()
        return auc_scores, mean_auc
    except ValueError as e:
        print(f"AUC calculation failed: {e}")
        return None, None


def main():
    set_seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_files, val_files = split_by_patient(CSV_PATH, train_ratio=0.8, seed=RANDOM_SEED)
    
    train_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=train_transform, sample_filter=train_files, validate=False)
    val_dataset = ChestXRayDataset(CSV_PATH, IMAGE_DIRS, transform=val_transform, sample_filter=val_files, validate=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    train_labels = np.array([train_dataset.label_dict[img] for img in train_dataset.samples])
    pos_weights = compute_pos_weights(torch.tensor(train_labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    
    model = build_baseline_cnn(num_classes=14).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
        
        auc_scores, mean_auc = evaluate(model, val_loader, device)
        if mean_auc:
            print(f"Mean AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
