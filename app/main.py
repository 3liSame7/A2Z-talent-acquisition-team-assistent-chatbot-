import os
from process_cv import process_resumes
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

def main():
    st.title("Resume Processing System")
    st.markdown("Upload resumes to process and store in FAISS.")

    uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf"])
    upload_dir = "uploaded_resumes"

    if uploaded_files:
        os.makedirs(upload_dir, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(upload_dir, file.name), "wb") as f:
                f.write(file.read())

        if st.button("Process Resumes"):
            process_resumes(upload_dir)
            st.success("Resumes processed and stored in FAISS.")

if __name__ == "__main__":
    main()
