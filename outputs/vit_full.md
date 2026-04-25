Random seed set to 42
Using device: cuda
GPU: Tesla T4
Training model: vit
ViT strategy: full
📁 Using KAGGLE paths:
   CSV: /kaggle/working/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/dataset/labels.csv
   Images: 12 directories
✅ All paths verified!

Loading and splitting data...
Patient-based split: 24644 train patients, 6161 val patients
Image split: 89703 train images, 22417 val images
Using 20000 train / 4000 val samples...
Aligned samples: 112120
Train dataset size: 20000
Val dataset size: 4000

Building VIT model (Strategy: full)...
config.json: 100%|█████████████████████████████| 502/502 [00:00<00:00, 2.28MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors: 100%|██████████████████████| 346M/346M [00:03<00:00, 107MB/s]
Loading weights: 100%|█| 198/198 [00:00<00:00, 1338.40it/s, Materializing param=
ViTForImageClassification LOAD REPORT from: google/vit-base-patch16-224-in21k
Key                 | Status     | 
--------------------+------------+-
pooler.dense.weight | UNEXPECTED | 
pooler.dense.bias   | UNEXPECTED | 
classifier.weight   | MISSING    | 
classifier.bias     | MISSING    | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Trainable params: 85809422
Epoch 1/12 - Loss: 1.3008 - Time: 707.58s - GPU Memory: 4694.77 MB
Mean AUC: 0.7036, Macro AUC: 0.7036
F1: 0.1304, Precision: 0.0768, Recall: 0.6723
Per-class AUC: [0.70680442 0.70206805 0.74893392 0.87025175 0.76886337 0.65797229
 0.69951268 0.69626754 0.68790768 0.63059748 0.6322048  0.68910968
 0.71518968 0.64487704]
💾 New best model saved to /kaggle/working/outputs/best_model_vit_full.pth

Epoch 2/12 - Loss: 1.2049 - Time: 719.13s - GPU Memory: 4694.77 MB
Mean AUC: 0.7423, Macro AUC: 0.7423
F1: 0.1436, Precision: 0.0849, Recall: 0.6550
Per-class AUC: [0.74363213 0.88315621 0.75402995 0.87427944 0.80798424 0.75591302
 0.72999508 0.6366483  0.71444698 0.68629965 0.59930701 0.73166202
 0.71679546 0.75793132]
💾 New best model saved to /kaggle/working/outputs/best_model_vit_full.pth

Epoch 3/12 - Loss: 1.1259 - Time: 719.91s - GPU Memory: 4694.77 MB
Mean AUC: 0.7528, Macro AUC: 0.7528
F1: 0.1566, Precision: 0.0943, Recall: 0.7171
Per-class AUC: [0.73165187 0.84552997 0.75832357 0.85485377 0.80241796 0.79938364
 0.77257071 0.6540268  0.6925128  0.708519   0.66663447 0.70747124
 0.81934035 0.72664029]
💾 New best model saved to /kaggle/working/outputs/best_model_vit_full.pth

Epoch 4/12 - Loss: 1.0706 - Time: 720.29s - GPU Memory: 4694.77 MB
Mean AUC: 0.7612, Macro AUC: 0.7612
F1: 0.1687, Precision: 0.1041, Recall: 0.6894
Per-class AUC: [0.74472695 0.86003657 0.75731771 0.87583573 0.80486786 0.8124252
 0.77652579 0.64572896 0.71032537 0.75566671 0.66268513 0.70311173
 0.80774324 0.73936221]
💾 New best model saved to /kaggle/working/outputs/best_model_vit_full.pth

Epoch 5/12 - Loss: 1.0094 - Time: 720.97s - GPU Memory: 4694.77 MB
Mean AUC: 0.7697, Macro AUC: 0.7697
F1: 0.1799, Precision: 0.1087, Recall: 0.6610
Per-class AUC: [0.76233385 0.89533182 0.77237467 0.85796947 0.82704396 0.81168841
 0.75915809 0.63348572 0.71249658 0.75719036 0.69275836 0.72700957
 0.81988503 0.74665069]
💾 New best model saved to /kaggle/working/outputs/best_model_vit_full.pth

Epoch 6/12 - Loss: 0.9568 - Time: 718.82s - GPU Memory: 4694.77 MB
Mean AUC: 0.7613, Macro AUC: 0.7613
F1: 0.1876, Precision: 0.1190, Recall: 0.5375
Per-class AUC: [0.76368294 0.89526173 0.76509115 0.85975921 0.83758204 0.80194931
 0.77001404 0.56751002 0.7116443  0.75312922 0.65856492 0.69974032
 0.82205491 0.75268595]
⏳ No improvement for 1 epochs

Epoch 7/12 - Loss: 0.9055 - Time: 721.24s - GPU Memory: 4694.77 MB
Mean AUC: 0.7636, Macro AUC: 0.7636
F1: 0.1930, Precision: 0.1315, Recall: 0.4903
Per-class AUC: [0.75681276 0.88409541 0.75950195 0.82813842 0.8280549  0.8034603
 0.75503885 0.67888903 0.69880166 0.7607992  0.65952477 0.706076
 0.81529985 0.75632762]
⏳ No improvement for 2 epochs

Epoch 8/12 - Loss: 0.8366 - Time: 718.62s - GPU Memory: 4694.77 MB
Mean AUC: 0.7598, Macro AUC: 0.7598
F1: 0.1868, Precision: 0.1170, Recall: 0.5520
Per-class AUC: [0.75975562 0.89510753 0.76146973 0.84508958 0.83273728 0.80082356
 0.75951074 0.5123998  0.70936851 0.74783959 0.69066333 0.75002855
 0.81064671 0.76203975]
⏳ No improvement for 3 epochs
🛑 Early stopping triggered after 8 epochs
🎉 Training completed! Best AUC: 0.7697
📁 All outputs saved to /kaggle/working/outputs/