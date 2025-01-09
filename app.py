import streamlit as st
import os
from uuid import uuid4
from filelock import FileLock
from rag_qa import rag_pipeline, BASE_CHROMA_DB_DIR, CHROMA_LOCK_FILE

def main():
    st.title("Q&A Application Using RAG")

    # Input for uploading PDF file
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file with a unique name
        file_name = f"uploaded_{uuid4().hex}.pdf"
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully!")

        # Unique directory for Chroma DB for each session
        unique_db_dir = os.path.join(BASE_CHROMA_DB_DIR, f"session_{uuid4().hex}/")
        os.makedirs(unique_db_dir, exist_ok=True)

        # Prevent multiple pipeline initializations using session state
        try:
            if "chain" not in st.session_state:
                with FileLock(CHROMA_LOCK_FILE):  # Avoid simultaneous access to the database
                    st.session_state.chain = rag_pipeline(file_name, unique_db_dir)
        except RuntimeError as e:
            st.error(f"Failed to initialize the RAG pipeline: {e}")
            os.remove(file_name)  # Clean up the uploaded file
            return

        if st.session_state.chain:
            st.info("RAG Pipeline completed. You can now interact with the document.")
            
            # User query input
            query = st.text_input("Ask a question related to the document:")

            if query:
                with st.spinner("Processing your query..."):
                    try:
                        response = st.session_state.chain({"question": query})
                        st.write(f"Answer: {response['answer']}")
                    except Exception as e:
                        st.error(f"An error occurred while querying the document: {e}")
        else:
            st.error("Failed to initialize the RAG pipeline.")
        
        # Clean up the uploaded file after processing
        os.remove(file_name)
        st.info("Uploaded file has been cleaned up to save disk space.")
    else:
        st.info("Please upload a PDF file to begin.")

if __name__ == "__main__":
    main()
