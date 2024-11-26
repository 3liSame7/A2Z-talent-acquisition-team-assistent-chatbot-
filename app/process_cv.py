import os
from tika import parser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import load_or_create_faiss_index, insert_into_faiss

def extract_text_with_pypdf2(file_path):
    """Extract text from PDF using PyPDF2."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_resumes(directory):
    files = os.listdir(directory)
    text_data = []

    for file in files:
        file_path = os.path.join(directory, file)
        content = None
        try:
            # Primary method: PyPDF2
            content = extract_text_with_pypdf2(file_path)
            if not content:
                # Fallback to Tika if PyPDF2 fails
                print(f"PyPDF2 failed for {file}, trying Tika.")
                parsed = parser.from_file(file_path)
                content = parsed.get("content", "").strip()
            if content:
                text_data.append(content)
                print(f"Extracted content from {file}:\n{content[:500]}")  # Log first 500 characters
            else:
                print(f"No valid text found in {file}.")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not text_data:
        print("No valid text data found in the resumes.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [splitter.split_text(text) for text in text_data]
    all_chunks = [chunk for sublist in chunks for chunk in sublist]

    print(f"Generated {len(all_chunks)} chunks from resumes.")

    dim = 384  # Match the embedding dimension
    index = load_or_create_faiss_index(dim)
    metadata = [{"id": i, "text": chunk} for i, chunk in enumerate(all_chunks)]
    print("Generated Metadata:", metadata)

    insert_into_faiss(index, all_chunks, metadata)
    print("Resumes processed and stored in FAISS.")

