import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import numpy as np
import evaluate
from datasets import Dataset


with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

texts = [f"Question: {item['instruction']} Answer: {item['response']}" for item in qa_data]

dataset = Dataset.from_dict({"text": texts})

checkpoint = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.map(
    lambda batch: {"labels": batch["input_ids"]},
    batched=True
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="output",
    push_to_hub=False,
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=0.0005
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=None
)


trainer.train()

model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_model")

qa_pipeline = pipeline(
    "text-generation",
    model="qa_model",
    tokenizer="qa_model",
    device=0 if torch.cuda.is_available() else -1
)

while True:
    user_q = input("Ask a question (or type 'quit' to exit): ")
    if user_q.lower() == "quit":
        break

    prompt = f"Question:{user_q} Answer:"
    output = qa_pipeline(prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.7,repetition_penalty=1.2,eos_token_id=tokenizer.eos_token_id)
    print("Model:", output[0]["generated_text"])
