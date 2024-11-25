from tika import parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import load_or_create_faiss_index, insert_into_faiss
import os

def process_resumes(directory):
    # List all files in the given directory
    files = os.listdir(directory)
    text_data = []

    # Parse each file using Tika
    for file in files:
        file_path = os.path.join(directory, file)
        
        # Use Tika to extract the content of each file
        try:
            parsed = parser.from_file(file_path)
            # Append only if there's content in the parsed file
            if parsed.get("content"):
                text_data.append(parsed["content"])
            else:
                print(f"Warning: No content found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Chunk the parsed text using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [splitter.split_text(text) for text in text_data]
    
    # Flatten the list of chunks (since splitter generates a list of lists)
    all_chunks = [chunk for sublist in chunks for chunk in sublist]

    # Insert chunks into FAISS index
    dim = 384  # This should match the embedding dimension you plan to use (e.g., SentenceTransformer "all-MiniLM-L6-v2" has dim=384)
    
    # Load or create FAISS index
    index = load_or_create_faiss_index(dim)
    
    # Create metadata for each chunk
    metadata = [{"id": i, "text": chunk} for i, chunk in enumerate(all_chunks)]
    
    # Insert chunks and metadata into FAISS
    insert_into_faiss(index, all_chunks, metadata)

    print("Resumes processed and stored in FAISS.")

