import json
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "EleutherAI/pythia-70m"

TRAIN_FILE = "data/student_train.jsonl"

OUTPUT_DIR = "models/pythia-70m-command-parser"

MAX_LENGTH = 256

BATCH_SIZE = 4
NUM_EPOCHS = 3

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

SEED = 42

# Log every N batches.
LOG_EVERY = 25


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("PYTHIA-70M STUDENT KNOWLEDGE DISTILLATION")
print("=" * 70)

print(f"Model        : {MODEL_NAME}")
print(f"Training file: {TRAIN_FILE}")
print(f"Output       : {OUTPUT_DIR}")
print("Examples     : loading...")
print(f"Max length   : {MAX_LENGTH}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay : {WEIGHT_DECAY}")
print(f"Device       : {DEVICE}")
print(f"Seed         : {SEED}")

print("=" * 70)


# ============================================================
# Load records
# ============================================================

records = load_jsonl(TRAIN_FILE)

print(f"\nLoaded training examples: {len(records)}")

if len(records) != 1500:
    raise ValueError(
        f"Expected exactly 1500 training examples, "
        f"but found {len(records)}."
    )


# Validate records.

for index, record in enumerate(records):

    if "text" not in record:
        raise ValueError(
            f"Example {index} is missing 'text'."
        )

    if "target" not in record:
        raise ValueError(
            f"Example {index} is missing 'target'."
        )


# ============================================================
# Hugging Face dataset
# ============================================================

dataset = Dataset.from_list(records)

print(f"Dataset created: {len(dataset)} examples")


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"PAD token: {tokenizer.pad_token}")


# ============================================================
# Tokenization
#
# The model receives:
#
# ### Command:
# <command>
#
# ### Response:
# <teacher-generated JSON>
#
# Only the response tokens contribute to the loss.
# ============================================================

def tokenize_example(example):

    prompt = example["text"]
    target = example["target"]

    full_text = prompt + target

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # Make sure target tokens remain after truncation.
    if len(input_ids) <= prompt_length:

        raise ValueError(
            "The target was completely truncated "
            "for one training example."
        )

    # Ignore prompt tokens.
    labels = [-100] * len(input_ids)

    # Train only on teacher response.
    for i in range(
        prompt_length,
        len(input_ids)
    ):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)

print(
    f"Tokenization complete: "
    f"{len(tokenized_dataset)} examples"
)


# ============================================================
# Data collator
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt",
)


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)


steps_per_epoch = len(dataloader)

total_steps = (
    steps_per_epoch * NUM_EPOCHS
)

print("\nTraining configuration:")
print(f"Examples per epoch : {len(records)}")
print(f"Steps per epoch    : {steps_per_epoch}")
print(f"Total steps        : {total_steps}")


# ============================================================
# Load model
# ============================================================

print("\nLoading Pythia-70M student model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)

model.to(DEVICE)

print("Student model loaded.")
print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Training
# ============================================================

model.train()

training_start = time.time()

epoch_losses = []
global_step = 0


print("\n")
print("=" * 70)
print("STARTING CPU TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    epoch_loss = 0.0

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        global_step += 1

        # Move tensors to CPU.
        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        # Clear gradients.
        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # Backpropagation.
        loss.backward()

        # Update weights.
        optimizer.step()

        loss_value = loss.item()

        epoch_loss += loss_value

        # Logging.
        if (
            batch_index == 1
            or batch_index % LOG_EVERY == 0
            or batch_index == steps_per_epoch
        ):

            elapsed = (
                time.time()
                - training_start
            )

            avg_loss = (
                epoch_loss / batch_index
            )

            print(
                f"  Step "
                f"{batch_index:4d}/{steps_per_epoch} "
                f"| Global "
                f"{global_step:4d}/{total_steps} "
                f"| Loss "
                f"{loss_value:.4f} "
                f"| Avg "
                f"{avg_loss:.4f} "
                f"| Time "
                f"{elapsed / 60:.1f} min"
            )

    # Epoch average.
    average_epoch_loss = (
        epoch_loss / steps_per_epoch
    )

    epoch_losses.append(
        average_epoch_loss
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    print(
        f"\nEpoch {epoch + 1} complete."
    )

    print(
        f"Average loss: "
        f"{average_epoch_loss:.4f}"
    )

    print(
        f"Epoch time: "
        f"{epoch_time / 60:.2f} minutes"
    )


# ============================================================
# Training finished
# ============================================================

total_training_time = (
    time.time()
    - training_start
)

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Total training time: "
    f"{total_training_time / 60:.2f} minutes"
)

for i, loss in enumerate(
    epoch_losses,
    start=1,
):

    print(
        f"Epoch {i} loss: "
        f"{loss:.4f}"
    )


# ============================================================
# Save model
# ============================================================

print("\nSaving trained student model...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"Model saved to: "
    f"{OUTPUT_DIR}"
)


# ============================================================
# Save training metrics
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

metrics = {
    "model_name": MODEL_NAME,
    "training_examples": len(records),
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "device": str(DEVICE),
    "steps_per_epoch": steps_per_epoch,
    "total_steps": total_steps,
    "epoch_losses": epoch_losses,
    "total_training_time_seconds": total_training_time,
    "total_training_time_minutes": (
        total_training_time / 60
    ),
    "final_epoch_loss": epoch_losses[-1],
    "torch_version": torch.__version__,
    "transformers_version": (
        __import__(
            "transformers"
        ).__version__
    ),
    "datasets_version": (
        __import__(
            "datasets"
        ).__version__
    ),
    "numpy_version": np.__version__,
}


metrics_path = (
    "results/student_training_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# Save configuration
# ============================================================

config = {
    "model_name": MODEL_NAME,
    "train_file": TRAIN_FILE,
    "training_examples": len(records),
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "device": "cpu",
    "torch_version": torch.__version__,
    "transformers_version": (
        __import__(
            "transformers"
        ).__version__
    ),
    "datasets_version": (
        __import__(
            "datasets"
        ).__version__
    ),
    "numpy_version": np.__version__,
}


config_path = (
    "results/student_training_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# Final message
# ============================================================

print("\n")
print("=" * 70)
print("STUDENT TRAINING ARTIFACTS")
print("=" * 70)

print(
    f"Model directory : "
    f"{OUTPUT_DIR}"
)

print(
    f"Metrics         : "
    f"{metrics_path}"
)

print(
    f"Configuration   : "
    f"{config_path}"
)

print("=" * 70)
