# AGENTS.md

## Project Overview
Medical imaging research project evaluating Vision Transformer (ViT) adaptation strategies for multi-label chest X-ray disease classification. Compares CNN baselines (DenseNet-121) against ViT fine-tuning approaches (full, partial, PEFT/LoRA).

## Environment Setup

### Dependencies
```
pip install torch torchvision transformers peft scikit-learn pandas matplotlib numpy Pillow
```

### Hardware
- CUDA-enabled GPU recommended; falls back to CPU

## Commands

### Python/Linting
```bash
# Syntax check
python3 -m py_compile <file.py>

# Format code
black <file.py>          # Single file
black .                  # Entire project

# Lint
flake8 <file.py>

# Auto-sort imports
isort <file.py>
```

### Testing
```bash
pytest                         # Run all tests
pytest tests/                  # Specific directory
pytest tests/test_model.py     # Single test file
pytest -v                      # Verbose output
pytest -k "test_name"          # Tests matching pattern
pytest --cov=.                 # With coverage
```

### Jupyter Notebook
```bash
# Validate notebook syntax
python3 -c "import nbformat; nbformat.read('notebook.ipynb', as_version=4)"

# Execute notebook
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

### ML Training
```bash
python3 train.py --strategy full --epochs 10 --batch_size 32
python3 train.py --strategy peft_lora
python3 train.py --strategy partial
```

## Code Style Guidelines

### Python Version
- Target Python 3.9+

### General Style
- Follow PEP 8; line length: 100 chars; 4 spaces (no tabs)

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
import json

# Third-party
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Local
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
```

### Type Annotations
```python
def build_vit_model(strategy: str, num_classes: int = 14) -> nn.Module:
    ...
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
- HuggingFace ViT returns object with `.logits` attribute
- Torchvision models return logits directly

### LoRA/PEFT
- Target modules: query, value (attention layers)
- Ensure classifier head is trainable: `modules_to_save=["classifier"]`
- Use `model.print_trainable_parameters()` to verify trainable vs frozen params

### Dataset Implementation
```python
class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        # Return (image_tensor, label_tensor)
        ...
```

## File Organization
```
myPorject/
├── ViT_XRay_Roadmap.ipynb    # Main research notebook
├── ViT_XRay_Roadmap.md       # Project description
├── requirements.txt          # Dependencies
├── src/                      # Python source code (when refactored)
├── tests/                    # Unit tests
├── data/                     # Dataset files
├── models/                   # Saved model checkpoints
└── AGENTS.md                 # This file
```

## Common Tasks

### Adding a New Model
1. Implement model builder function with snake_case name
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
