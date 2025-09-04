import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline

checkpoint = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

qa_pipeline = pipeline(
    "text-generation",
    model=checkpoint,
    tokenizer=checkpoint,
    device=0 if torch.cuda.is_available() else -1
)

while True:
    user_q = input("Ask a question (or type 'quit' to exit): ")
    if user_q.lower() == "quit":
        break

    prompt = f"Question:{user_q} Answer:"
    output = qa_pipeline(prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.7,repetition_penalty=1.2,eos_token_id=tokenizer.convert_tokens_to_ids("<END>"))
    print("Model:", output[0]["generated_text"])