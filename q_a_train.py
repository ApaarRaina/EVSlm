import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import numpy as np
import evaluate
from datasets import Dataset

# Load dataset
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

texts = [f"Question: {item['instruction']} Answer: {item['response']} <END>" for item in qa_data]

dataset = Dataset.from_dict({"text": texts})

checkpoint = "qa_model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Add <END> if not present
if "<END>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': ["<END>"]})

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
tokenized_dataset = tokenized_dataset.map(lambda batch: {"labels": batch["input_ids"]}, batched=True)

# Train args
training_args = TrainingArguments(
    output_dir="output",
    push_to_hub=False,
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=0.0005
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=None
)
trainer.train()

# Save fine-tuned model
model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_model")

# Load pipeline
qa_pipeline = pipeline(
    "text-generation",
    model="qa_model",
    tokenizer="qa_model",
    device=0 if torch.cuda.is_available() else -1
)

# Interactive Q&A loop
conversation_history = ""   # keep conversation separate

while True:
    user_q = input("Ask a question (or type 'quit' to exit): ")
    if user_q.lower() == "quit":
        break

    # Append conversation
    conversation_history += f"\nQuestion: {user_q} Answer:"

    output = qa_pipeline(
        conversation_history,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.convert_tokens_to_ids("<END>")
    )

    answer = output[0]["generated_text"]
    print("Model:", answer)


    conversation_history += " " + answer.replace("<END>", "").strip()