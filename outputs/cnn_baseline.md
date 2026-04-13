Random seed set to 42
Using device: cuda
GPU: Tesla T4
📁 Using KAGGLE paths:
   CSV: /kaggle/working/Evaluating-Vision-Transformer-Adaptation-for-X-Ray-Disease-Classification/dataset/labels.csv
   Images: 12 directories
✅ All paths verified!

Loading and splitting data...
Patient-based split: 24644 train patients, 6161 val patients
Image split: 89703 train images, 22417 val images
Using 10000 train / 2000 val samples...
Aligned samples: 112120
Train dataset size: 10000
Val dataset size: 2000

Building model...
Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
100%|███████████████████████████████████████| 30.8M/30.8M [00:00<00:00, 180MB/s]
Trainable params: 6968206
Epoch 1/10 - Loss: 1.2387 - Time: 135.07s
Mean AUC: 0.7313, Macro AUC: 0.7313
F1: 0.1646, Precision: 0.1033, Recall: 0.5048
Per-class AUC: [0.70113795 0.75838591 0.71348933 0.86676254 0.78510329 0.74599066
 0.71367753 0.99382407 0.65224489 0.6021134  0.65406436 0.55786802
 0.73482372 0.75855548]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_1.pth

Epoch 2/10 - Loss: 1.0961 - Time: 122.64s
Mean AUC: 0.7651, Macro AUC: 0.7651
F1: 0.1610, Precision: 0.0963, Recall: 0.6594
Per-class AUC: [0.72434305 0.78293833 0.70879449 0.89830158 0.82723099 0.80378194
 0.7652888  0.97529628 0.68100659 0.58531787 0.6913967  0.60976311
 0.83362555 0.82417366]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_2.pth

Epoch 3/10 - Loss: 1.0025 - Time: 121.83s
Mean AUC: 0.7712, Macro AUC: 0.7712
F1: 0.1750, Precision: 0.1094, Recall: 0.6147
Per-class AUC: [0.72510573 0.80156863 0.71207522 0.89647133 0.8248511  0.85184908
 0.78351888 0.96327825 0.69698086 0.6647079  0.66175279 0.63187817
 0.79993284 0.78243124]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_3.pth

Epoch 4/10 - Loss: 0.9318 - Time: 122.02s
Mean AUC: 0.7803, Macro AUC: 0.7803
F1: 0.1782, Precision: 0.1110, Recall: 0.6451
Per-class AUC: [0.730388   0.83180449 0.7355494  0.92482923 0.83758224 0.8446981
 0.77669276 0.98397596 0.69434698 0.67774055 0.67433319 0.53226734
 0.88211119 0.79762922]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_4.pth

Epoch 5/10 - Loss: 0.8663 - Time: 121.28s
Mean AUC: 0.7786, Macro AUC: 0.7786
F1: 0.1831, Precision: 0.1136, Recall: 0.5501
Per-class AUC: [0.72695046 0.8249048  0.74882943 0.91891362 0.83250785 0.80204656
 0.76682378 0.98898348 0.67469655 0.6760567  0.70008543 0.54045685
 0.87254244 0.82676567]
⏳ No improvement for 1 epochs

Epoch 6/10 - Loss: 0.8055 - Time: 121.36s
Mean AUC: 0.7881, Macro AUC: 0.7881
F1: 0.2108, Precision: 0.1324, Recall: 0.6030
Per-class AUC: [0.72550496 0.83678318 0.73707035 0.91906614 0.84334816 0.82575848
 0.78239926 0.99699549 0.69093457 0.72929553 0.70137048 0.5677665
 0.8537437  0.82281559]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_6.pth

Epoch 7/10 - Loss: 0.7304 - Time: 121.31s
Mean AUC: 0.7778, Macro AUC: 0.7778
F1: 0.2473, Precision: 0.1972, Recall: 0.5166
Per-class AUC: [0.72084632 0.84278488 0.71270371 0.89656938 0.83336448 0.84831847
 0.77315329 0.95443165 0.66875213 0.68904639 0.71194299 0.53604061
 0.8755973  0.82626918]
⏳ No improvement for 1 epochs

Epoch 8/10 - Loss: 0.6704 - Time: 121.72s
Mean AUC: 0.7884, Macro AUC: 0.7884
F1: 0.2226, Precision: 0.1457, Recall: 0.6007
Per-class AUC: [0.72186505 0.85522023 0.72512271 0.90775784 0.84320487 0.81818862
 0.76055747 0.98597897 0.69602589 0.73015464 0.70834337 0.61467005
 0.85322663 0.81675538]
💾 New best model saved to /kaggle/working/outputs/best_model_epoch_8.pth

Epoch 9/10 - Loss: 0.6157 - Time: 121.70s
Mean AUC: 0.7772, Macro AUC: 0.7772
F1: 0.2427, Precision: 0.1771, Recall: 0.4788
Per-class AUC: [0.71882813 0.82637113 0.71867438 0.89627523 0.81775194 0.81622883
 0.77172667 0.997997   0.67509309 0.68410653 0.70519645 0.58666667
 0.8650063  0.80111566]
⏳ No improvement for 1 epochs

Epoch 10/10 - Loss: 0.5586 - Time: 121.45s
Mean AUC: 0.7667, Macro AUC: 0.7667
F1: 0.2524, Precision: 0.1858, Recall: 0.4794
Per-class AUC: [0.70078553 0.86140381 0.70223303 0.90696256 0.84123617 0.72957932
 0.76896371 0.98397596 0.65757814 0.65498282 0.67806424 0.60527919
 0.83792257 0.80552574]
⏳ No improvement for 2 epochs

🎉 Training completed! Best AUC: 0.7884
📁 All outputs saved to /kaggle/working/outputs/