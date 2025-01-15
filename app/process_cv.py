import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from vector_store import load_or_create_faiss_index, insert_into_faiss, generate_embeddings

# Path to your local SentenceTransformer model
model_path = r"/mnt/d/Users/ziad.mahmoud/djangotest/project/last_rag/genai_chatbot/models/stella_en_400M_v5"

def extract_text_with_pypdf2(file_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text.strip()

def extract_metadata(full_text):
    """
    Simple regex-based metadata extractor.
    Adapt the patterns to fit your actual CV format/fields.
    """
    name = re.search(r"Name:\s*(.*)", full_text)
    email = re.search(r"Email:\s*(\S+@\S+)\s*", full_text)
    phone = re.search(r"Phone:\s*(\+?[0-9\-()\s]+)", full_text)

    return {
        "name": name.group(1).strip() if name else None,
        "email": email.group(1).strip() if email else None,
        "phone": phone.group(1).strip() if phone else None,
        "text": full_text
    }

def process_resumes(directory, model_path=model_path):
    """
    Process all PDF resumes in the `directory` by:
      1) extracting text
      2) generating embeddings
      3) inserting into a FAISS index
    """
    files = os.listdir(directory)
    text_data = []
    metadata_list = []

    for file in files:
        if not file.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(directory, file)
        try:
            resume_text = extract_text_with_pypdf2(file_path)
            if resume_text:
                metadata = extract_metadata(resume_text)
                text_data.append(resume_text)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not text_data:
        print("No valid PDF resumes found in the directory.")
        return

    # Generate embeddings using the local model
    embeddings = generate_embeddings(text_data)

    # Create or load existing FAISS index
    dim = embeddings.shape[1]  # embedding dimension
    index = load_or_create_faiss_index(dim)

    # Insert the new embeddings + metadata into the FAISS index
    insert_into_faiss(index, embeddings, metadata_list)

    print("Resumes processed successfully and stored in FAISS.")
# file: process_cv.py
