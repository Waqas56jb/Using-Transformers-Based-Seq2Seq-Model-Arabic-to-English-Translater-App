import torch
import pickle
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import Dataset

# Load dataset (limit to 100 samples)
dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "ara-eng")
train_data = dataset["test"].select(range(100))  # Use only first 100 samples
val_data = dataset["validation"].select(range(100))

# Load tokenizer and model
model_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Custom Dataset class
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]["sourceString"]
        tgt_text = self.data[idx]["targetString"]
        src_encoded = self.tokenizer(src_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        tgt_encoded = self.tokenizer(tgt_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": src_encoded["input_ids"].squeeze(0),
            "attention_mask": src_encoded["attention_mask"].squeeze(0),
            "labels": tgt_encoded["input_ids"].squeeze(0),
        }

# Create dataset instances
train_dataset = TranslationDataset(train_data, tokenizer)
val_dataset = TranslationDataset(val_data, tokenizer)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Training arguments (reduce epochs & batch size)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,  # Reduce batch size
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=2,  # Reduce epochs
    logging_dir="./logs",
    logging_steps=5,  # Log more frequently
    save_total_limit=1,
    predict_with_generate=True,
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save model
with open("nmt_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and saved as nmt_model.pkl")