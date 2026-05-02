This project investigates how to adapt Vision Transformers for multi-label chest X-ray disease classification. It compares multiple fine-tuning strategies (linear, partial, full, and LoRA-based) against CNN baselines, while addressing key challenges such as class imbalance and domain shift. The study includes advanced techniques like Grad-CAM for interpretability, per-class AUROC evaluation, scaling to larger ViT models, and self-supervised pretraining with DINO. The goal is to identify efficient and effective strategies for applying transformer models to medical imaging tasks.

## what is used

add PEFT (LoRA)
analyze compute vs performance
include interpretability (Grad-CAM)
use AUROC per class

### Systematic comparison of:
    partial vs full vs PEFT (LoRA, adapters)

### Tradeoff:
    performance vs compute vs data efficiency

### Practical question:
    “What’s the cheapest way to get near-SOTA performance?”.