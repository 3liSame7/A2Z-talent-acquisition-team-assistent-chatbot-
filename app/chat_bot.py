import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

class ResumeRAGSystem:
    def __init__(
        self,
        model_path=r"/mnt/d/Users/ziad.mahmoud/djangotest/project/last_rag/genai_chatbot/models/stella_en_400M_v5",
        index_path="resume_index"
    ):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device="cpu"
        )

        # Load or check for FAISS index
        self.index = self.load_faiss_index(index_path)

        # Load metadata
        self.metadata = self.load_metadata()

    def load_faiss_index(self, index_path):
        if not os.path.exists(index_path):
            print(f"[DEBUG] FAISS index file not found at {index_path}")
            return None
        index = faiss.read_index(index_path)
        print("[DEBUG] FAISS index loaded successfully.")
        return index

    def load_metadata(self, metadata_path="metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Metadata file not found at {metadata_path}")
            return []

    # def search_candidates(self, query, top_k=5):
    #     # Check if index is loaded
    #     if self.index is None:
    #         print("[ERROR] FAISS index not loaded. Please process resumes first.")
    #         return []

    #     # Encode the query to get its embedding
    #     embedding = self.embedding_model.encode([query])
    #     embedding = np.array(embedding).reshape(1, -1)

    #     # Perform the similarity search
    #     distances, indices = self.index.search(embedding, top_k)

    #     # Prepare the results
    #     results = []
    #     for i in range(len(indices[0])):
    #         idx = int(indices[0][i])
    #         if idx != -1 and idx < len(self.metadata):
    #             text = self.metadata[idx].get("text", "No text available")
    #             results.append({
    #                 "score": float(distances[0][i]),
    #                 "metadata": self.metadata[idx],
    #                 "text": text
    #             })
    #     return results
    def search_candidates(self, query, top_k=5):
        if self.index is None:
            print("[ERROR] FAISS index not loaded")
            return []

        # Stronger embedding with normalization
        embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        embedding = np.array(embedding).reshape(1, -1)

        # Perform similarity search
        distances, indices = self.index.search(embedding, top_k * 2)  # Fetch more results for re-ranking

        # Filter out irrelevant results based on similarity threshold
        relevance_threshold = 0.75  # Adjust as needed
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            score = float(distances[0][i])
            if idx != -1 and idx < len(self.metadata) and score >= relevance_threshold:
                text = self.metadata[idx].get('text', 'No text available')
                results.append({
                    "score": score,
                    "metadata": self.metadata[idx],
                    "text": text
                })
            if len(results) >= top_k:  # Stop after top_k relevant results
                break

        return results


    def generate_response(self, query):
        # Search for relevant candidates
        results = self.search_candidates(query, top_k=5)

        if not results:
            return "No relevant information found."

        # Build a simple context from the top retrieved candidates
        context = "\n\n".join(
            f"Candidate {i + 1}:\n{res['text']}"
            for i, res in enumerate(results)
        )

        # Prepare the prompt for Ollama
        # full_prompt = f"""
        # Context:
        # {context}

        # Question: {query}

        # Instructions:
        # - Carefully analyze the context.
        # - Provide a precise and relevant answer.
        # - If information is insufficient, state "Information not available".

        # Answer:
        # """
        full_prompt = f"""
        You are an intelligent assistant helping recruiters analyze candidate resumes to identify the best-fit candidates for job roles.

        ### Context:
        Here are the details of the top matching candidates based on the search:
        {context}

        ### User Query:
        "{query}"

        ### Instructions:
        1. Carefully analyze the user's query and the candidate context provided.
        2. Identify and prioritize candidates who best match the user's requested skills, experience, or qualifications.
        3. Clearly explain why each recommended candidate is suitable, highlighting relevant skills or projects.
        4. Avoid repeating the candidate's full context—summarize only the most relevant points.
        5. If information is insufficient, politely respond with: "No sufficient data available for this request."
        6. Format the response using bullet points or numbering for readability.
        7. Keep the tone professional, clear, and concise.

        ### Response:
        """


        # Call Ollama using the prompt
        try:
            response = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant skilled in extracting information from CVs."
                    },
                    {"role": "user", "content": full_prompt}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"[ERROR] Failed to generate response: {str(e)}")
            return f"An error occurred: {str(e)}"


# Create a global instance for easy import
rag_system = ResumeRAGSystem()

def process_user_question(user_question):
    """
    Wrapper function to process user questions using the RAG system.
    """
    return rag_system.generate_response(user_question)
