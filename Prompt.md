"Please update kaggle_train.py to act as a unified master training script using argparse.

Currently, kaggle_train.py is hardcoded to train our CNN baseline. I want to be able to train either the CNN or various Vision Transformer (ViT) strategies from the command line, while preserving all the robust tracking, metrics (F1, Precision, Recall), full checkpointing, and early stopping mechanisms we already implemented.

Here are the specific changes to make to kaggle_train.py:

1. Import argparse at the top of the file.

2. Import the model builder functions from our source module: from src.train import build_baseline_cnn, build_vit_model.

3. Inside the main() block (or before the training setup), set up an ArgumentParser to accept two arguments:

--model: type string, choices ['cnn', 'vit'], default 'vit'

--strategy: type string, choices ['full', 'partial', 'peft_lora'], default 'full'

4. Find the line where the model is currently initialized and replace it with conditional logic:

If args.model == 'cnn', initialize with model = build_baseline_cnn(num_classes=14).

If args.model == 'vit', initialize with model = build_vit_model(strategy=args.strategy, num_classes=14).

5. Ensure the instantiated model is still properly moved to the correct hardware device (e.g., model = model.to(device)).

6. Crucial: Update the save paths for the outputs so we don't overwrite previous runs. Modify the paths for history.json and the model checkpoint .pth file to dynamically include the model and strategy names (e.g., f'/kaggle/working/outputs/history_{args.model}_{args.strategy}.json' and f'/kaggle/working/outputs/best_model_{args.model}_{args.strategy}.pth').

Please implement these changes without removing or altering the existing training loop logic, metric calculations, or early stopping logic."