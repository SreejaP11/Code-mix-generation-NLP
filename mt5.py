import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Settings
model_name = "google/mt5-small"
dataset_name = "findnitai/english-to-hinglish"
batch_size = 4
num_epochs = 1
learning_rate = 5e-4
max_length = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")
print("Original dataset length:", len(dataset))

# Filter valid examples
def is_valid(example):
    return (
        "translation" in example and
        "en" in example["translation"] and
        "hi_ng" in example["translation"] and
        example["translation"]["en"] and
        example["translation"]["hi_ng"]
    )

dataset = dataset.filter(is_valid)
print("Filtered dataset length:", len(dataset))

# Subset for debugging
dataset = dataset.select(range(100000))
print("Subset dataset length:", len(dataset))

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing
def preprocess(example):
    input_text = "translate English to Hinglish: " + example["translation"]["en"]
    target_text = example["translation"]["hi_ng"]

    inputs = tokenizer(input_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    targets = tokenizer(target_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    input_ids = inputs["input_ids"].squeeze()
    attention_mask = inputs["attention_mask"].squeeze()
    labels = targets["input_ids"].squeeze()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Dataset wrapper
class HinglishDataset(Dataset):
    def __init__(self, dataset):
        self.data = [preprocess(ex) for ex in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = HinglishDataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop with progress bar
print("\nStarting training loop...")
model.train()

# Wrap the epoch loop with tqdm for progress bar
for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
    total_loss = 0
    # Wrap the batch loop with tqdm for progress bar
    for i, batch in enumerate(tqdm(train_loader, desc=f"Batch Progress (Epoch {epoch+1})", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Print updates only every 50th batch
        if i % 50 == 0:
            print(f"\nBatch {i+1}")
            print(f"Loss: {total_loss / (i+1):.4f}")  # Average loss so far

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / (i + 1)
    print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

# Save the model
print("\nSaving model...")
model.save_pretrained("mt5-simple-hinglish-debug")
tokenizer.save_pretrained("mt5-simple-hinglish-debug")
print("✅ Model saved!")
