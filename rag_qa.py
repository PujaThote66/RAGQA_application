import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from uuid import uuid4
from filelock import FileLock
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Constants
BASE_CHROMA_DB_DIR = "db1/"
CHROMA_LOCK_FILE = "db1.lock"

# Ensure base Chroma DB directory exists
os.makedirs(BASE_CHROMA_DB_DIR, exist_ok=True)

def reset_chroma_db(db_dir):
    """
    Resets the Chroma database directory to ensure no dimensionality mismatch or conflicts.
    """
    try:
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)
        os.makedirs(db_dir, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to reset Chroma DB: {e}")

def rag_pipeline(pdf_path, db_dir):
    """
    The RAG pipeline to process a PDF, split it into chunks, generate embeddings, 
    and initialize a Retrieval QA chain with a Multi-Query Retriever and custom prompt.
    """
    try:
        # Reset Chroma DB to avoid dimensionality mismatch
        reset_chroma_db(db_dir)

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        # Embeddings using Hugging Face
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Vector Store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_dir
        )

        # Multi-Query Retriever
        llm = OpenAI()  # This LLM generates query variations
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llm
        )

       
        prompt = PromptTemplate(
                        template="""
            Provide an answer to the question: "{question}" based on the context provided below. 
            If the answer is not present in the document, reply with: "I don't know. The information is not given in the document."

            Context:
            {context}

            Answer:
            """,
                input_variables=["question", "context"]
            )

        # Retrieval QA chain with custom prompt
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"  # Map retrieved documents to 'context'
            }
        )

        return chain

    except Exception as e:
        st.error(f"An error occurred in the RAG pipeline: {e}")
        return None

# # Streamlit UI
# def main():
#     st.title("Q&A Application Using RAG")

#     # Input for uploading PDF file
#     uploaded_file = st.file_uploader("Upload PDF", type="pdf")

#     if uploaded_file is not None:
#         # Save the uploaded file with a unique name
#         file_name = f"uploaded_{uuid4().hex}.pdf"
#         with open(file_name, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         st.success("File uploaded successfully!")

#         # Unique directory for Chroma DB for each session
#         unique_db_dir = os.path.join(BASE_CHROMA_DB_DIR, f"session_{uuid4().hex}/")
#         os.makedirs(unique_db_dir, exist_ok=True)

#         # Prevent multiple pipeline initializations using session state
#         if "chain" not in st.session_state:
#             with FileLock(CHROMA_LOCK_FILE):  # Avoid simultaneous access to the database
#                 st.session_state.chain = rag_pipeline(file_name, unique_db_dir)

#         if st.session_state.chain:
#             st.info("RAG Pipeline completed. You can now interact with the document.")
            
#             # User query input
#             query = st.text_input("Ask a question related to the document:")

#             if query:
#                 with st.spinner("Processing your query..."):
#                     try:
#                         response = st.session_state.chain({"question": query})
#                         st.write(f"Answer: {response['answer']}")
#                         st.write("Sources:")
#                         for doc in response["source_documents"]:
#                             st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
#                     except Exception as e:
#                         st.error(f"An error occurred while querying the document: {e}")
#         else:
#             st.error("Failed to initialize the RAG pipeline.")
#     else:
#         st.info("Please upload a PDF file to begin.")

# if __name__ == "__main__":
#     main()
