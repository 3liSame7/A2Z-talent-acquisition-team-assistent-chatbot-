# Resume Retrieval-Augmented Generation (RAG) System

This repository contains the implementation of a Resume Retrieval-Augmented Generation (RAG) system, designed to assist recruiters in analyzing candidate resumes and identifying the best-fit candidates for job roles.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [File Descriptions](#file-descriptions)
- [How It Works](#how-it-works)
  
## Overview

The Resume RAG system leverages a combination of FAISS indexing and a SentenceTransformer embedding model to store and retrieve candidate resumes based on user queries. Using the RAG framework, it retrieves relevant candidates' resumes and generates a natural language response summarizing the most suitable candidates for the recruiter.

## Features

- **Resume Processing**:
  - Extract text and metadata from resumes in PDF format.
  - Generate embeddings for resume text using the SentenceTransformer model.
- **FAISS Indexing**:
  - Store and manage resume embeddings for efficient similarity searches.
  - Perform vector similarity search to find relevant resumes for user queries.
- **RAG Framework**:
  - Retrieve relevant resumes and generate meaningful responses.
  - Use natural language models (Ollama) for prompt-based response generation.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/3liSame7/A2Z-talent-acquisition-team-assistent-chatbot-.git
   cd <repo_name>
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the SentenceTransformer model (`stella_en_400M_v5`) is available at the specified path in the code or download it.


5. Ensure the LLM model(`llama 3.2:1b`) is available at the specified path in the code or download it.


## Requirements

The following Python packages are required to run the project:

- `streamlit`
- `langchain`
- `langchain_community`
- `langchain_core`
- `python-dotenv`
- `langchain-huggingface`
- `langchain-ollama`
- `unstructured[pdf]`
- `onnx==1.16.1`
- `faiss-cpu==1.9.0.post1`
- `PyPDF2==3.0.1`

Install them using:

```bash
pip install -r requirements.txt
```

## File Descriptions

- **`process_cv.py`**: Contains functions for extracting text from resumes, generating embeddings, and updating the FAISS index.
- **`vector_store.py`**: Handles FAISS index creation, loading, and search functionality.
- **`metadata.json`**: Stores metadata of processed resumes.
- **`resume_index`**: The FAISS index file containing resume embeddings.

## How It Works

1. **Resume Processing**:
   - Extract text using PyPDF2.
   - Parse metadata (e.g., name, email, phone) using regex.
   - Generate embeddings with the `stella_en_400M_v5` model.
   - Store embeddings and metadata in FAISS.

2. **Query Handling**:
   - Encode the user query into an embedding vector.
   - Perform similarity search in the FAISS index.
   - Retrieve relevant candidates and their metadata.

3. **Response Generation**:
   - Use Ollama (e.g., `llama3.2:1b`) to generate a human-readable response summarizing candidate suitability based on the query.

