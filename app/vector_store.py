import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Path to the FAISS index and model
INDEX_PATH = "resume_index"
MODEL_PATH = r"/mnt/d/Users/ziad.mahmoud/djangotest/project/last_rag/genai_chatbot/models/stella_en_400M_v5"

# Initialize the embedding model
embedding_model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cpu")

def load_or_create_faiss_index(expected_dim):
    """Load or create a FAISS index with the expected dimension."""
    if os.path.exists(INDEX_PATH):
        print("[DEBUG] Loading existing FAISS index.")
        index = faiss.read_index(INDEX_PATH)
        # Validate the dimension of the index
        if index.d != expected_dim:
            print(f"[WARNING] FAISS index dimension {index.d} does not match expected dimension {expected_dim}. Recreating the index.")
            os.remove(INDEX_PATH)  # Remove old index
            index = faiss.IndexFlatIP(expected_dim)
            faiss.write_index(index, INDEX_PATH)  # Save the newly created index
            print("[DEBUG] New FAISS index created and saved with dimension:", expected_dim)
    else:
        index = faiss.IndexFlatIP(expected_dim)
        faiss.write_index(index, INDEX_PATH)  # Save the newly created index
        print("[DEBUG] New FAISS index created and saved with dimension:", expected_dim)
    return index

def generate_embeddings(text_chunks):
    """Generate embeddings for the given text chunks."""
    if not text_chunks:
        raise ValueError("No text chunks provided for embedding.")
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    return np.array(embeddings)

def insert_into_faiss(index, embeddings, metadata_list):
    """
    Insert embeddings and their metadata into the FAISS index.
    Each metadata in `metadata_list` should have at least a 'text' field (the raw resume text).
    """
    if len(embeddings.shape) != 2 or embeddings.shape[1] != index.d:
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match FAISS index dimension {index.d}.")
    if len(metadata_list) != embeddings.shape[0]:
        raise ValueError("The number of metadata entries does not match the number of embeddings.")

    # Merge metadata with the ID and embedding vectors
    valid_metadata = []
    for i in range(len(metadata_list)):
        meta = {
            "id": i,
            "embedding": embeddings[i].tolist()
        }
        meta.update(metadata_list[i])
        valid_metadata.append(meta)

    # Save all metadata to JSON
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(valid_metadata, f, indent=4)

    print("[DEBUG] Metadata saved successfully.")

    # Add embeddings to the index
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)  # Save updated index
    print("[DEBUG] FAISS index updated and saved.")
    print(f"[DEBUG] FAISS index now contains {index.ntotal} entries.")

def search_candidates(query, index, top_k=5):
    """
    Search for the top-k candidates in the FAISS index that match the given query.
    """
    # Generate embedding for the user query
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).reshape(1, -1)  # ensure 2D

    # Safety check
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {index.d}."
        )

    # Perform vector search
    distances, indices = index.search(query_embedding, top_k)
    print("[DEBUG] Search results - distances:", distances, "indices:", indices)

    # Load saved metadata
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Build results
    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        if idx != -1 and idx < len(metadata):
            candidate_text = metadata[idx].get('text', 'No text available')
            results.append({
                "score": float(distances[0][i]),
                "metadata": metadata[idx],
                "text": candidate_text
            })

    return results
