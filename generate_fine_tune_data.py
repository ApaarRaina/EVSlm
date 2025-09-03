import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
import json
import re
from pathlib import Path
from tqdm import tqdm



def pdf_to_text(pdf_path):
    """Extract text from PDF file"""
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
    """Split text into chunks of specified word count"""
    if not text:
        return []


    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        # Only add chunks that have substantial content
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    return chunks


def generate_qa_pairs(chunks, model_name="microsoft/Phi-3-mini-4k-instruct"):
    """Generate Q&A pairs from text chunks using transformer model"""
    if not chunks:
        print("No chunks to process")
        return []

    use_gpu = torch.cuda.is_available()
    print(f"Using device: {'GPU' if use_gpu else 'CPU'}")
    print(f"Processing {len(chunks)} chunks...")

    try:
        print("Loading model and tokenizer...")
        with tqdm(total=2, desc="Model Loading", unit="component") as pbar:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            pbar.update(1)

            if use_gpu:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
            pbar.update(1)

        print("Creating pipeline...")
        if use_gpu:
            qa_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False
            )
        else:
            qa_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                return_full_text=False
            )

        qa_pairs = []

        # Process chunks with progress bar
        for i in tqdm(range(len(chunks)), desc="Generating Q&A", unit="chunk"):
            chunk = chunks[i]


            prompt = f"""Based on the following text, create one clear and specific question and provide a comprehensive answer.

Text: {chunk}

Format your response exactly as:
Question: [your question here]
Answer: [your answer here]"""

            try:
                # Generate response
                output = qa_pipeline(
                    prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']

                # IMPROVED PARSING with better error handling
                question = ""
                answer = ""

                # Look for Question: and Answer: patterns
                if "Question:" in output and "Answer:" in output:
                    try:
                        question_part = output.split("Question:")[1].split("Answer:")[0].strip()
                        answer_part = output.split("Answer:")[1].strip()

                        # Clean up the extracted text
                        question = question_part.strip()
                        answer = answer_part.strip()

                        # Remove any trailing text after newlines in answer
                        if '\n' in answer:
                            answer = answer.split('\n')[0].strip()

                    except IndexError:
                        pass
                else:
                    # Fallback: try to extract Q&A from the generated text
                    lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
                    if len(lines) >= 2:
                        # Assume first line is question, second is answer
                        potential_question = lines[0]
                        potential_answer = lines[1]

                        if '?' in potential_question:
                            question = potential_question
                            answer = potential_answer

                # Clean and validate Q&A pair
                if question and answer and len(question) > 10 and len(answer) > 10:
                    # Ensure question ends with ?
                    if not question.endswith('?'):
                        question = question.rstrip('.!') + '?'

                    qa_pairs.append({
                        "instruction": question,
                        "response": answer,
                        "source_chunk": i
                    })
                else:
                    tqdm.write(f"Failed to extract valid Q&A from chunk {i + 1}")
                    tqdm.write(f"Generated text: {output[:200]}...")

            except Exception as e:
                tqdm.write(f"Error processing chunk {i + 1}: {e}")
                continue

        return qa_pairs

    except Exception as e:
        print(f"Error initializing model: {e}")
        return []


def save_alpaca_format(qa_pairs, output_file):
    """Save Q&A pairs in Alpaca format"""
    # Remove source_chunk info for final dataset
    clean_qa_pairs = []
    for pair in qa_pairs:
        clean_qa_pairs.append({
            "instruction": pair["instruction"],
            "response": pair["response"]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_qa_pairs, f, ensure_ascii=False, indent=2)


def process_data_folder(data_folder="data", output_file="qa_dataset.json", chunk_size=400):
    """Process all PDF files in the data folder"""
    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"Data folder '{data_folder}' not found!")
        return

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{data_folder}' folder!")
        return

    print(f"Found {len(pdf_files)} PDF files")

    all_qa_pairs = []

    # Process each PDF file with progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        tqdm.write(f"\nProcessing: {pdf_file.name}")

        # Extract text from PDF
        text = pdf_to_text(pdf_file)
        if not text:
            tqdm.write(f"No text extracted from {pdf_file.name}")
            continue

        tqdm.write(f"Extracted {len(text)} characters from {pdf_file.name}")

        # Split into chunks
        chunks = chunk_text(text, chunk_size=chunk_size)
        if not chunks:
            tqdm.write(f"No valid chunks created from {pdf_file.name}")
            continue

        tqdm.write(f"Created {len(chunks)} chunks")

        # Generate Q&A pairs
        qa_pairs = generate_qa_pairs(chunks)

        if qa_pairs:
            tqdm.write(f"Generated {len(qa_pairs)} Q&A pairs from {pdf_file.name}")
            all_qa_pairs.extend(qa_pairs)
        else:
            tqdm.write(f"No Q&A pairs generated from {pdf_file.name}")

    if all_qa_pairs:
        print("\nSaving dataset...")
        save_alpaca_format(all_qa_pairs, output_file)
        print(f"Total Q&A pairs generated: {len(all_qa_pairs)}")
        print(f"Dataset saved to: {output_file}")
    else:
        print("No Q&A pairs were generated from any PDF files")



if __name__ == "__main__":
    DATA_FOLDER = "data"
    OUTPUT_FILE = "qa_dataset.json"
    CHUNK_SIZE = 100

    # Process all PDFs in data folder
    process_data_folder(DATA_FOLDER, OUTPUT_FILE, CHUNK_SIZE)