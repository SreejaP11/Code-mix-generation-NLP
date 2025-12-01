import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os

print("Step 0: Checking GPU availability...")
if not torch.cuda.is_available():
    print("❌ CUDA not available! Please check your GPU setup. Run `nvidia-smi` to verify.")
    exit()
else:
    print("✅ CUDA is available. GPU:", torch.cuda.get_device_name(0))

# Step 1: Load dataset
print("Step 1: Loading dataset...")
dataset = load_dataset("findnitai/english-to-hinglish")

# Step 2: Load tokenizer and model name (T5)
print("Step 2: Loading tokenizer...")
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Configure QLoRA (4-bit quantization)
print("Step 3: Configuring QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Step 4: Load model in 4-bit
print("Step 4: Loading model...")
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically handle device assignment, especially with multiple GPUs
)

# Step 5: Add LoRA adapter
print("Step 5: Adding LoRA adapter...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, peft_config)

# Step 6: Preprocessing function
print("Step 6: Defining preprocessing function...")

def preprocess(sample):
    source_texts = ["Translate English to Hinglish: " + ex["en"] for ex in sample["translation"]]
    target_texts = [ex["hi_ng"] for ex in sample["translation"]]

    model_inputs = tokenizer(source_texts, max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(target_texts, max_length=128, padding="max_length", truncation=True)

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 7: Preprocess dataset
print("Step 7: Preprocessing dataset...")
train_dataset = dataset["train"].map(preprocess, batched=True, remove_columns=["translation"])

# Step 8: Data collator
print("Step 8: Creating data collator...")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Step 9: Define training arguments
print("Step 9: Defining training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="t5-hinglish-qlora",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=1e-5,
    save_steps=500,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=100,
    report_to="none"
)

# Step 10: Initialize Trainer
print("Step 10: Initializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 11: Start training
print("Step 11: Starting training...")
trainer.train()

# Step 12: Save model
print("Step 12: Saving model...")
trainer.save_model("t5-hinglish-qlora")
tokenizer.save_pretrained("t5-hinglish-qlora")

print("✅ Training and saving complete!")