🔴 1. This WILL break your code

Inside build_vit_model:

print(f"Trainable params: {count_trainable_params(model)}")

👉 Problem:

count_trainable_params is NOT defined anywhere in this file

💥 Result:

NameError: name 'count_trainable_params' is not defined
✅ Fix (do ONE of these)
Option A (clean)

Add this at the top:

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
Option B (lazy but fine)

Delete that print line

👉 My advice: keep it + define the function
This is useful for debugging strategies.

🟡 2. Your model saving name is misleading
save_path = f"best_model_cnn.pth"

But you're using:

model = build_vit_model(...)


Fix:
save_path = f"best_model_vit.pth"
🟡 3. You’re always using FULL fine-tuning
model = build_vit_model(strategy="full", num_classes=14)

So:

your partial and peft_lora code is never tested

👉 Not wrong, but:

you think you built multiple strategies
but you're only running one
🟡 4. DataLoader optimization (small but real)

You used:

pin_memory=True

Good 👍

But better:

pin_memory=torch.cuda.is_available()
🟡 5. Subtle research-level issue (important)

You compute:

preds_binary = (all_preds > 0.5)

👉 Fixed threshold = 0.5

For medical datasets:

this is usually NOT optimal
classes are imbalanced

💡 Real pros:

tune threshold per class

But for now, it's fine.