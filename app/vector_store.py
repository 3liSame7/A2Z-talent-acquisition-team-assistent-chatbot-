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

def generate_embeddings(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided for embedding.")
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    return np.array(embeddings)


def insert_into_faiss(index, text_chunks, metadata_list):
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]
    if not valid_chunks:
        print("No valid text chunks for embedding.")
        return

    embeddings = generate_embeddings(valid_chunks)

    # Check if embeddings are 2D
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings should be 2D, but got shape: {embeddings.shape}")

    # Ensure the metadata length matches the number of embeddings
    valid_metadata = [
        {"id": i, "text": valid_chunks[i]} for i in range(len(valid_chunks))
    ]
    if len(valid_metadata) != len(embeddings):
        raise ValueError(
            f"Mismatch between metadata and embeddings: {len(valid_metadata)} metadata items for {len(embeddings)} embeddings."
        )

    # Save metadata
    print("Saving Metadata to metadata.json...")
    with open("E:\\Headway\\Tasks\\genai_chatbot\\app\\metadata.json", "w") as f:
        json.dump(valid_metadata, f, indent=4)
    print("Metadata successfully saved!")

    # Insert embeddings into FAISS
    print(f"Adding {len(embeddings)} embeddings to FAISS index.")
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
