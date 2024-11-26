import os
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import faiss
import json
import numpy as np

# Constants
INDEX_PATH = "resume_index"

# Initialize SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None
    index = faiss.read_index(INDEX_PATH)
    return index

# Search FAISS
def search_candidates(query, index, top_k=5):
    embedding = embedding_model.encode([query])  # Create embedding for the query
    distances, indices = index.search(np.array(embedding), top_k)  # Search FAISS

    # Load metadata for results
    with open("E:\\Headway\\Tasks\\genai_chatbot\\app\\metadata.json", "r") as f:
        metadata = json.load(f)  # This is a list of dictionaries, not a dictionary

    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        if idx != -1 and idx < len(metadata):  # Ensure the index is valid
            result_metadata = metadata[idx]  # Access list by index
            results.append({
                "score": float(distances[0][i]),
                "metadata": result_metadata,
            })
    return results


# Build conversational chain
def get_conversational_chain():
    prompt_template = """
    Based on the provided context, answer the question as clearly and precisely as possible.
    If the answer cannot be found in the context, reply: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chat_model = ChatOpenAI(temperature=0.3)
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    return chain

# Process user question
def process_user_question(user_question):
    index = load_faiss_index()
    if not index:
        return "No FAISS index found. Process resumes first."

    results = search_candidates(user_question, index, top_k=5)
    if not results:
        return "No relevant candidates found."

    response = "\n\n".join(
        [
            f"Candidate {i+1}:\n{res['metadata'].get('text', 'No text available')} (Score: {res['score']})"
            for i, res in enumerate(results)
        ]
    )
    return response
