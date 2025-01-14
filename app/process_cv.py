import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from vector_store import load_or_create_faiss_index, insert_into_faiss

model_path = r"D:\\Users\\ziad.mahmoud\\last_rag\\genai_chatbot\\genai_chatbot\\models\\stella_en_400M_v5"

def extract_text_with_pypdf2(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text.strip()

def extract_metadata(text):
    name = re.search(r"Name:\s*(.*)", text)
    email = re.search(r"Email:\s*(\S+@\S+)\s*", text)
    phone = re.search(r"Phone:\s*(\+?[0-9\-()\s]+)", text)
    return {
        "name": name.group(1) if name else None,
        "email": email.group(1) if email else None,
        "phone": phone.group(1) if phone else None,
    }

def process_resumes(directory, model_path=model_path):
    files = os.listdir(directory)
    text_data, metadata_list = [], []

    for file in files:
        file_path = os.path.join(directory, file)
        try:
            text = extract_text_with_pypdf2(file_path)
            if text:
                metadata = extract_metadata(text)
                metadata['text'] = text
                text_data.append(text)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not text_data:
        print("No valid text data found in the resumes.")
        return

    embedding_model = SentenceTransformer(model_path, device="cpu", trust_remote_code=True)
    dummy_embedding = embedding_model.encode(["dummy text"], show_progress_bar=False)
    dim = dummy_embedding.shape[1]
    embeddings = embedding_model.encode(text_data, show_progress_bar=True)

    index = load_or_create_faiss_index(dim)
    insert_into_faiss(index, embeddings, metadata_list)

    print("Resumes processed successfully and embeddings stored in FAISS.")
