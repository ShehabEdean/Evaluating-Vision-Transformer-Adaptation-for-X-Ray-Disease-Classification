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
├── ViT_XRay_Roadmap.md       # Project description
├── README.md                # This file
├── requirements.txt         # Dependencies
├── AGENTS.md                # Agent instructions
├── data/                    # Dataset files (images_001-012/, train_val_list.txt, test_list.txt, PDFs)
├── dataset/
│   └── labels.csv          # Labels CSV
├── src/
│   ├── dataset.py          # ChestXRayDataset + alignment validation
│   ├── transforms.py        # Image transforms (train/val)
│   └── train.py            # Training loop, model builders, evaluation
├── kaggle_train.py         # Kaggle training script
├── outputs/                # Model outputs/results
├── Related work/           # Reference papers
└── venv/                   # Virtual environment
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
- **Metrics**: AUC (mean and macro)

### Data Processing

- **Transforms**: Resize, horizontal flips, rotation, color jitter, normalization
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

## Results

The project compares:
- Training time and resource usage
- Model performance (AUC - mean and macro)
- Parameter efficiency
- Generalization capabilities

## Citation

@mastersthesis{benbertal_zenikheri_2026,
  title        = {Vision Transformer Adaptation for Chest X-Ray Disease Classification},
  author       = {Benbertal, Ali Chihab Edine and Zenikheri, Saad},
  year         = {2026},
  school       = {Amar Telidji University of Laghouat},
  type         = {Master's Thesis},
  address      = {Laghouat, Algeria},
  note         = {Supervised by Dr. Younes Guellouma.}
}

## Contact

For questions or issues, please open a GitHub issue.