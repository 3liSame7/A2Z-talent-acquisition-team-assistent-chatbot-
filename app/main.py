import os
import streamlit as st
from process_cv import process_resumes
from chat_bot import process_user_question

def main():
    st.set_page_config(page_title="Resume Screening Chatbot", page_icon="👍")

    st.title("Resume Screening System")
    st.sidebar.title("Navigation")

    # Two sections: upload/resume processing, and chat interface
    option = st.sidebar.radio("Select Action", ["Process Resumes", "Chatbot", "Admin Tools"])

    upload_dir = "uploaded_resumes"
    os.makedirs(upload_dir, exist_ok=True)

    if option == "Process Resumes":
        st.header("Upload and Process Multiple Resumes")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF Resumes",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if uploaded_files:
            for file in uploaded_files:
                with open(os.path.join(upload_dir, file.name), "wb") as f:
                    f.write(file.read())

            if st.button("Process Resumes"):
                # Process the resumes in the chosen directory
                process_resumes(upload_dir)
                st.success("Resumes processed and stored in FAISS.")

    elif option == "Chatbot":
        st.header("Chat with the Resume Screening Bot")
        user_question = st.text_input("Enter your question:")
        
        if st.button("Ask"):
            if user_question.strip():
                response = process_user_question(user_question)
                st.write(response)
            else:
                st.warning("Please enter a valid question.")

    elif option == "Admin Tools":
        st.header("Administration Options")
        st.markdown("Use these tools to manage your FAISS index and metadata.")

        # Clear Index button
        if st.button("Clear Index"):
            index_path = "resume_index"
            metadata_file = "metadata.json"
            
            if os.path.exists(index_path):
                os.remove(index_path)
            
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            st.success("FAISS index and metadata have been cleared.")

if __name__ == "__main__":
    main()
