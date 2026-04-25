🧱 STEP 1 — Modify build_vit_model()

This is where everything happens.

Go to your src/train.py where build_vit_model is defined.

✍️ Add this logic:
def build_vit_model(strategy="full", num_classes=14):
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        problem_type="multi_label_classification"
    )

    if strategy == "partial":
        print("Applying partial fine-tuning...")

        # Freeze ALL layers first
        for param in model.vit.parameters():
            param.requires_grad = False

        # 🔥 Unfreeze LAST encoder block
        for param in model.vit.encoder.layer[-1].parameters():
            param.requires_grad = True

        # 🔥 Unfreeze classification head
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif strategy == "full":
        print("Using full fine-tuning")

    return model
🧠 Why this works

You are allowing:

high-level features to adapt
low-level features to stay stable

👉 Best trade-off

🧪 STEP 2 — Verify it’s working (CRITICAL)

You already print:

print(f"Trainable params: {count_trainable_params(model)}")
✅ Expected result:
Strategy	Params
Full	~86M
Partial	~5–10M

👉 If it still shows ~86M:

❌ your freezing failed

⚙️ STEP 3 — Adjust optimizer (IMPORTANT)

Right now:

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
🔥 Change to:
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

👉 This ensures:

only trainable params are updated
avoids wasting compute
⚙️ STEP 4 — Keep everything else SAME

DO NOT change:

dataset
split
transforms
batch size
metrics
scheduler

👉 This is what makes your comparison valid.

🚀 STEP 5 — Run it

Command:

python kaggle_train.py --model vit --strategy partial
📊 STEP 6 — What to expect
Metric	Expected
AUC	0.74 – 0.77
Time	↓ faster
Memory	↓ lower
Stability	↑ better

👉 If it beats full tuning:

🔥 major insight

🧠 What you’re testing (THIS IS YOUR PAPER)

You are now comparing:

Strategy	AUC	Params	Time
Full	0.73	86M	high
Partial	?	~7M	lower

👉 This answers:

“Do we need full fine-tuning?”

⚔️ Brutal truth

If partial gets:

same AUC
less compute

👉 Full fine-tuning becomes inefficient