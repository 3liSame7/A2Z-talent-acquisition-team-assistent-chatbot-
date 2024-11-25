import os
import streamlit as st
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
        st.warning("FAISS index not found. Please process resumes first.")
        return None
    index = faiss.read_index(INDEX_PATH)
    return index

# Search FAISS
def search_candidates(query, index, top_k=5):
    embedding = embedding_model.encode([query])
    D, I = index.search(np.array(embedding), top_k)

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    results = [{"score": float(D[0][i]), "metadata": metadata.get(int(I[0][i]), {})} for i in range(len(I[0])) if I[0][i] != -1]
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
        return

    results = search_candidates(user_question, index, top_k=5)
    if not results:
        st.warning("No relevant candidates found.")
        return

    # Build context from search results
    context = "\n\n".join([f"Candidate {i+1}:\n{res['metadata']}" for i, res in enumerate(results)])
    chain = get_conversational_chain()
    response = chain({"context": context, "question": user_question}, return_only_outputs=True)
    st.success("Reply:")
    st.write(response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Screening Chatbot", page_icon=":robot:")
    st.title("📄 Resume Screening Chatbot")
    st.markdown("Ask about candidates based on their resumes!")

    # User question input
    user_question = st.text_input("Enter your query about candidates:")
    if st.button("Ask"):
        if user_question.strip():
            process_user_question(user_question)
        else:
            st.warning("Please enter a valid question.")

if __name__ == "__main__":
    main()
