import os
import torch
import re
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader

# ---------- PDF utils ----------
def pdf_to_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=400):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

# ---------- Model init (load ONCE) ----------
def init_model(model_name="microsoft/Phi-3-mini-4k-instruct"):
    use_gpu = torch.cuda.is_available()
    print(f"Using device: {'GPU' if use_gpu else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # pad_token -> eos_token if missing (no new tokens added!)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None
    )

    # IMPORTANT: do not add new tokens (no vocab resize); rely on tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Please choose a model with a defined EOS token.")

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,   # only return completion
        # NOTE: no `device=` arg when using accelerate
    )
    return tokenizer, text_gen, eos_token_id

# ---------- Q/A generator ----------
_QA_RE = re.compile(
    r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=(?:\n+---|\n+Question:|$))",
    flags=re.S | re.I
)

def build_prompt(chunk, num_questions=5):
    # Let the model decide answer length naturally; just ask for diversity.
    return f"""Based on the text below, generate exactly {num_questions} diverse Question-Answer pairs.
Provide a mix of factual (short), explanatory (medium), and reasoning (long) answers as needed by the question.
Do NOT include anything except pairs in the format shown. Separate pairs by a line with '---'.

Text:
{chunk}

Format STRICTLY:
Question: <question text>
Answer: <answer text>
---
"""

def parse_qa_block(generated_text):
    pairs = []
    for m in _QA_RE.finditer(generated_text.strip()):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a:
            if not q.endswith("?"):
                q = q.rstrip(".! ") + "?"
            # Strip trailing separators if any
            a = re.sub(r"\s*---\s*$", "", a).strip()
            pairs.append((q, a))
    return pairs

def generate_qa_pairs_for_chunks(chunks, text_gen, tokenizer, eos_token_id, num_questions=5, batch_size=4):
    qa_pairs = []
    eos_token_str = tokenizer.eos_token  # e.g., "</s>" or "<|end|>"

    # Build prompts in batches
    prompts = [build_prompt(c, num_questions=num_questions) for c in chunks]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Q&A", unit="batch"):
        batch_prompts = prompts[i:i+batch_size]

        # Let the model decide length. Use a generous cap; EOS will stop long ones early.
        outputs = text_gen(
            batch_prompts,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
            batch_size=len(batch_prompts)  # leverage GPU better
        )

        # HF pipeline returns a list of lists (one per prompt), take first sequence per prompt
        for j, out in enumerate(outputs):
            text = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            pairs = parse_qa_block(text)
            src_idx = i + j
            for q, a in pairs:
                # Append EOS token token-text so your SFT data teaches clean stopping
                qa_pairs.append({
                    "instruction": q,
                    "response": f"{a} {eos_token_str}",
                    "source_chunk": src_idx
                })

    return qa_pairs

# ---------- Save ----------
def save_alpaca_format(qa_pairs, output_file):
    clean = [{"instruction": p["instruction"], "response": p["response"]} for p in qa_pairs]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

# ---------- Orchestration ----------
def process_data_folder(
    data_folder="data",
    output_file="qa_dataset.json",
    chunk_size=400,
    model_name="microsoft/Phi-3-mini-4k-instruct",
    num_questions=5,
    batch_size=4
):
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Data folder '{data_folder}' not found!")
        return

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{data_folder}' folder!")
        return

    print(f"Found {len(pdf_files)} PDF files")

    tokenizer, text_gen, eos_token_id = init_model(model_name)

    all_chunks = []
    file_chunk_spans = []  # (file_name, start_idx, end_idx) if you want traceability

    for pdf_file in tqdm(pdf_files, desc="Reading PDFs", unit="file"):
        text = pdf_to_text(pdf_file)
        if not text:
            tqdm.write(f"No text extracted from {pdf_file.name}")
            continue
        chunks = chunk_text(text, chunk_size=chunk_size)
        if not chunks:
            tqdm.write(f"No valid chunks from {pdf_file.name}")
            continue
        start = len(all_chunks)
        all_chunks.extend(chunks)
        end = len(all_chunks)
        file_chunk_spans.append((pdf_file.name, start, end))
        tqdm.write(f"{pdf_file.name}: {len(text)} chars -> {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks to process.")
        return

    qa_pairs = generate_qa_pairs_for_chunks(
        all_chunks,
        text_gen=text_gen,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        num_questions=num_questions,
        batch_size=batch_size
    )

    if qa_pairs:
        print("\nSaving dataset...")
        save_alpaca_format(qa_pairs, output_file)
        print(f"Total Q&A pairs generated: {len(qa_pairs)}")
        print(f"Dataset saved to: {output_file}")
    else:
        print("No Q&A pairs were generated.")

if __name__ == "__main__":
    process_data_folder(
        data_folder="data",
        output_file="qa_dataset.json",
        chunk_size=400,                  # small chunks for tighter Q-A grounding
        model_name="microsoft/Phi-3-mini-4k-instruct",
        num_questions=5,                 # 5 QAs per chunk
        batch_size=4                     # tune for your GPU
    )