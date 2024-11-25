import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Initialize SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index path
INDEX_PATH = "resume_index"

# Load or create FAISS index
def load_or_create_faiss_index(dim):
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print("FAISS index loaded.")
    else:
        index = faiss.IndexFlatIP(dim)
        print("FAISS index created.")
    return index

# Generate Embeddings
def generate_embeddings(text_chunks):
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    return np.array(embeddings)

# Insert embeddings into FAISS
def insert_into_faiss(index, text_chunks, metadata_list):
    embeddings = generate_embeddings(text_chunks)
    metadata = {i: metadata_list[i] for i in range(len(metadata_list))}
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print("Embeddings and metadata inserted into FAISS.")

# Search FAISS index
def search_faiss(query, index, top_k=5):
    embedding = generate_embeddings([query])[0]
    D, I = index.search(np.array([embedding]), top_k)
    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    results = [{"score": float(D[0][i]), "metadata": metadata.get(int(I[0][i]), {})} for i in range(len(I[0])) if I[0][i] != -1]
    return results

if __name__ == "__main__":
    # Example usage
    dim = embedding_model.get_sentence_embedding_dimension()
    index = load_or_create_faiss_index(dim)

    # Sample data
    example_chunks = ["Python developer with AI experience.", "Data scientist specializing in ML."]
    example_metadata = [{"id": 1, "skills": ["Python", "AI"]}, {"id": 2, "skills": ["Data Science", "ML"]}]
    
    insert_into_faiss(index, example_chunks, example_metadata)

    query = "Looking for an AI expert."
    results = search_faiss(query, index)
    print("Search Results:", results)
