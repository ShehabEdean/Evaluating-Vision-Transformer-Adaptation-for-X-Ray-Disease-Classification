# Vision Transformer Adaptation for Chest X-Ray Disease Classification

## Project Overview

This project evaluates Vision Transformer (ViT) adaptation strategies for multi-label chest X-ray disease classification, comparing CNN baselines (DenseNet-121) against ViT fine-tuning approaches (full, partial, PEFT/LoRA).

## Dataset

**NIH ChestX-ray14** dataset with 14 disease labels:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Pleural_Thickening

## Environment Setup

```bash
pip install torch torchvision transformers peft scikit-learn pandas matplotlib numpy Pillow
```

- CUDA-enabled GPU recommended; falls back to CPU
- Python 3.9+

## Project Structure

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

## Key Features

### Model Architectures

1. **CNN Baseline**: DenseNet-121 with modified classifier head
2. **ViT Full Fine-tuning**: Complete fine-tuning of ViT-B/16
3. **ViT Partial Fine-tuning**: Freeze early layers, fine-tune later layers
4. **ViT with LoRA**: Parameter-efficient fine-tuning using Low-Rank Adaptation

### Training Configuration

- **Batch Size**: 16 (smaller for ViT to fit in GPU memory)
- **Learning Rate**: 5e-5 (lower for ViT fine-tuning)
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss Function**: BCEWithLogitsLoss with class weighting
- **Metrics**: AUC, F1, Precision, Recall

### Data Processing

- **Transforms**: Random resized crops, horizontal flips, normalization
- **Validation**: Image-label alignment verification
- **Splitting**: Patient-level split (80% train, 20% val)

## Usage

### Training

```bash
python src/train.py
```

### Kaggle Training

```bash
python kaggle_train.py
```

### Notebook Development

```bash
# Validate notebook syntax
python3 -c "import nbformat; nbformat.read('notebook.ipynb', as_version=4)"

# Execute notebook
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

## Results

The project compares:
- Training time and resource usage
- Model performance (AUC, F1, Precision, Recall)
- Parameter efficiency
- Generalization capabilities

## Code Quality

- Type annotations
- Google-style docstrings
- Comprehensive error handling
- Modular design for easy extension

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Citation

If you use this code or dataset in your research, please cite:

```
@article{your_citation_here,
  title={Vision Transformer Adaptation for Chest X-Ray Disease Classification},
  author={Your Name},
  year={2024}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
