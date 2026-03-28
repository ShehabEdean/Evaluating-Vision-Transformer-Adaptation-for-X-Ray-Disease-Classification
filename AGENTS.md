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
flake8 <file.py>                  # Lint
isort <file.py>                  # Sort imports
```

### Testing
```bash
pytest                         # Run all tests
pytest -v                      # Verbose
pytest -k "test_name"          # Match pattern
pytest --cov=.                 # With coverage
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
class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None, labels_filter: list = None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.labels_filter = labels_filter or DISEASE_LABELS
        self.image_names = self.df['Image Index'].values
        self._build_label_matrix()

    def _build_label_matrix(self) -> None:
        label_vectors = []
        for _, row in self.df.iterrows():
            findings = row['Finding Labels'].split('|')
            vector = [1.0 if disease in findings else 0.0 for disease in self.labels_filter]
            label_vectors.append(vector)
        self.labels = np.array(label_vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, labels
```

## File Organization
```
myPorject/
├── ViT_XRay_Roadmap.ipynb    # Main research notebook
├── ViT_XRay_Roadmap.md       # Project description
├── requirements.txt          # Dependencies
├── data/                     # Dataset files (images/, Data_Entry_2017.csv)
├── models/                   # Saved model checkpoints (when added)
└── AGENTS.md                 # This file
```

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
