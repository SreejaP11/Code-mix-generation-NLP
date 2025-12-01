import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 only

import torch
import random
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from tqdm.auto import tqdm
import os


model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "findnitai/english-to-hinglish"
output_dir = "./llama2-english-to-hinglish"
max_seq_length = 100
batch_size = 8
grad_accum_steps = 2
learning_rate = 4e-4
num_epochs = 1
val_split_size = 0.1
log_interval = 100  # Update progress bar every 100 steps

# Set HF token
HF_TOKEN = os.getenv("HF_TOKEN")

# Load and split dataset
full_dataset = load_dataset(dataset_name, split="train")

# Add this line to control dataset length (e.g., take first 10,000 samples)
full_dataset = full_dataset.select(range(min(len(full_dataset), 100000)))  # replace desired_length with your number

split_dataset = full_dataset.train_test_split(test_size=val_split_size)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
# Formatting and tokenization
def format_text(example):
    return {
        "text": (
            f"Translate English to Hinglish:\n\n"
            f"English: {example['translation']['en']}\n"
            f"Hinglish: {example['translation']['hi_ng']}"
        )
    }

train_dataset = train_dataset.map(format_text)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Model setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=True
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs // grad_accum_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)
scaler = torch.cuda.amp.GradScaler()

def generate_validation_example(step):
    """Generate and print a validation example"""
    model.eval()
    with torch.no_grad():
        idx = random.randint(0, len(val_dataset)-1)
        example = val_dataset[idx]['translation']
        
        english = example['en']
        prompt = f"Translate English to Hinglish:\n\nEnglish: {english}\nHinglish:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated.split("Hinglish:")[-1].strip()
        
        tqdm.write(f"\n--- Validation Example at Step {step} ---")
        tqdm.write(f"English: {english}")
        tqdm.write(f"Reference: {example['hi_ng']}")
        tqdm.write(f"Generated: {generated}\n")
    
    model.train()

# Training loop with progress bar
model.train()
global_step = 0

for epoch in range(num_epochs):
    epoch_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs}",
        total=len(train_loader),
        postfix={"loss": "N/A", "lr": learning_rate}
    )
    
    for batch in epoch_progress:
        # Training steps
        inputs = batch["input_ids"].to(model.device)
        masks = batch["attention_mask"].to(model.device)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (global_step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Update progress bar
        if global_step % log_interval == 0:
            epoch_progress.set_postfix({
                "loss": f"{loss.item() * grad_accum_steps:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Validation examples
        if global_step % 100 == 0 and global_step > 0:
            generate_validation_example(global_step)
        
        global_step += 1

    epoch_progress.close()

# Final save
model.save_pretrained(output_dir)