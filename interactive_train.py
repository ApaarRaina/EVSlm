import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Akshaykumarbm/oasst-english-openai-formate", split="train")

checkpoint = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Add special tokens
if "<END>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': ["<END>"]})

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32
)
model.resize_token_embeddings(len(tokenizer))

# Convert conversations into plain text
def conversation_to_text(example):
    parts = []
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{role.upper()}: {content}")
    return {"text": "\n".join(parts) + " <END>"}

dataset = dataset.map(conversation_to_text)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # increased to allow longer convos
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(lambda batch: {"labels": batch["input_ids"]}, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="output",
    push_to_hub=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # helps with small GPU
    num_train_epochs=3,             # start smaller, avoid overfitting
    learning_rate=5e-5,
    fp16=False, # faster training on most GPUs
    bf16=True,
    logging_steps=50,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_model")
