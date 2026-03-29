# AGENTS.md

## Project Overview

Medical imaging research project evaluating Vision Transformer (ViT) adaptation strategies for multi-label chest X-ray disease classification. Compares CNN baselines (DenseNet-121) against ViT fine-tuning approaches (full, partial, PEFT/LoRA).

**Dataset:** NIH ChestX-ray14 with 14 disease labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Pleural_Thickening.

## Environment Setup

```bash
pip install torch torchvision transformers peft scikit-learn pandas matplotlib numpy Pillow
```

- CUDA-enabled GPU recommended; falls back to CPU
- Python 3.9+

## Commands

### Notebook Development
```bash
# Validate notebook syntax
python3 -c "import nbformat; nbformat.read('notebook.ipynb', as_version=4)"

# Execute notebook
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

### Python/Linting
```bash
python3 -m py_compile <file.py>  # Syntax check
black <file.py>                  # Format code
ruff check <file.py>             # Lint (recommended)
ruff format <file.py>            # Format (alternative to black)
isort <file.py>                  # Sort imports
```

### Testing
```bash
pytest                         # Run all tests
pytest -v                      # Verbose
pytest -k "test_name"          # Match pattern
pytest --cov=.                 # With coverage
pytest tests/                  # Run specific test directory
pytest tests/test_file.py::test_function  # Run single test function
```

## Code Style Guidelines

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ChestXRayDataset` |
| Functions | snake_case | `build_baseline_cnn` |
| Variables | snake_case | `running_loss` |
| Constants | SCREAMING_SNAKE | `NUM_CLASSES` |
| Private methods | _leading_underscore | `_internal_method` |

### Imports (alphabetical within groups)
```python
# Standard library
import os

# Third-party
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Local/project
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
```

### Type Annotations
```python
def build_vit_model(strategy: str, num_classes: int = 14) -> nn.Module:
```

### Docstrings (Google style)
```python
def train_one_epoch(model, dataloader, optimizer, criterion):
    """Train model for one epoch.
    
    Args:
        model: The neural network model.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
    
    Returns:
        Average loss for the epoch.
    """
```

### Error Handling
```python
try:
    auc_scores = roc_auc_score(all_labels, all_preds, average=None)
except ValueError as e:
    print(f"AUC calculation failed: {e}")
    return None, None
```

## ML-Specific Guidelines

### PyTorch Conventions
- Use `torch.no_grad()` for inference
- Set `model.train()` / `model.eval()` appropriately
- Move tensors to device: `tensor.to(device)`
- Use `optimizer.zero_grad()` before backward pass

### Multi-Label Classification
- Use `BCEWithLogitsLoss` (includes sigmoid internally)
- Apply sigmoid only during evaluation for probability conversion
- Handle cases where classes may have zero positive/negative samples

### Vision Transformer Usage
- Standard input size: 224x224 or 384x384
- Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- HuggingFace ViT returns object with `.logits` attribute (extract via `outputs.logits`)
- Torchvision models return logits directly

### LoRA/PEFT
- Target modules: query, value (attention layers)
- Ensure classifier head is trainable: `modules_to_save=["classifier"]`
- Use `model.print_trainable_parameters()` to verify trainable vs frozen params

### Dataset Implementation
```python
from src.dataset import ChestXRayDataset, validate_alignment, scan_images, load_label_dict

# Alignment validation (run before training)
label_dict = load_label_dict("dataset/labels.csv")
path_cache = scan_images(["data/images_001/images", ...])
common_files, missing_images, missing_labels = validate_alignment(label_dict, path_cache)

# Dataset with built-in alignment
dataset = ChestXRayDataset(
    csv_file="dataset/labels.csv",
    image_dirs=["data/images_001/images", "data/images_002/images", ...],
    transform=train_transform,
    validate=True  # validates alignment on init
)
```

## File Organization
```
myPorject/
├── ViT_XRay_Roadmap.ipynb    # Main research notebook
├── ViT_XRay_Roadmap.md       # Project description
├── requirements.txt          # Dependencies
├── data/                     # Raw dataset (images_001-012/, Data_Entry_2017.csv)
├── dataset/
│   └── labels.csv           # Labels CSV (symlink to data/)
│   └── images/               # Image directory (or symlinks to data/)
├── src/
│   ├── dataset.py           # ChestXRayDataset + alignment validation
│   ├── transforms.py        # Image transforms (train/val)
│   └── train.py             # Training loop, model builders, evaluation
├── models/                   # Saved model checkpoints (when added)
└── AGENTS.md                 # This file
```

## Dataset Alignment

Always validate image-label alignment before training:
- Use `validate_alignment(label_dict, path_cache)` to verify matching
- Dataset builds `samples` list from intersection of labels and images
- Set `validate=False` to skip validation check (not recommended for production)

## Common Tasks

### Adding a New Model Strategy
1. Add builder function with snake_case name
2. Register in `build_vit_model()` with strategy parameter
3. Add corresponding test case

### Debugging Training
- Check trainable parameters: `model.print_trainable_parameters()`
- Monitor GPU memory: `torch.cuda.memory_allocated()`
- Verify data shapes: `print(images.shape, labels.shape)`
- Validate label distribution: `print(labels.sum(dim=0))`

## Notebook Conventions
- Use markdown cells for section headers and explanations
- Code cells should be self-contained
- Include `# TODO:` comments for incomplete implementations
- Document assumptions in comments (e.g., column ordering)
