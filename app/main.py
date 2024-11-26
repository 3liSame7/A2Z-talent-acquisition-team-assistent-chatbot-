import os
from process_cv import process_resumes
import streamlit as st
from chat_bot import process_user_question

def main():
    st.set_page_config(page_title="Resume Screening Chatbot", page_icon="📄")
    st.title("Resume Screening System")
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Action", ["Process Resumes", "Chatbot"])

    upload_dir = "uploaded_resumes"
    os.makedirs(upload_dir, exist_ok=True)

    if option == "Process Resumes":
        st.header("Upload and Process Resumes")
        uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf"])

        if uploaded_files:
            for file in uploaded_files:
                with open(os.path.join(upload_dir, file.name), "wb") as f:
                    f.write(file.read())
            if st.button("Process Resumes"):
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

if __name__ == "__main__":
    main()
