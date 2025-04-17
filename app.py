from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
import re
from dotenv import load_dotenv
import subprocess

# Install Playwright dependencies
def install_dependencies():
    with st.spinner("Setting up environment..."):
        subprocess.run(["playwright", "install", "chromium"], check=True)
        subprocess.run(["playwright", "install-deps"], check=True)

install_dependencies()

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

VECTORSTORE_PATH = "my_vector_db"

# Streamlit UI
st.title("🔎 Website QA & Contact Info Extractor")

url = st.text_input("Enter website URL:", placeholder="https://example.com")
query = st.text_area("Ask a question about the website:", height=200)

def get_custom_loader(url):
    return PlaywrightURLLoader(
        urls=[url],
        remove_selectors=["header", "footer"],
        continue_on_failure=False
    )

# ... [rest of your existing functions remain the same] ...

if st.button("Submit", type='primary'):
    if not url:
        st.warning("Please enter a website URL.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            try:
                # Load or build vectorstore
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                if os.path.exists(VECTORSTORE_PATH):
                    vectorstore = FAISS.load_local(
                        VECTORSTORE_PATH,
                        embedding_model,
                        allow_dangerous_deserialization=True
                    )
                else:
                    loader = get_custom_loader(url)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = url

                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    vectorstore = FAISS.from_documents(chunks, embedding_model)
                    vectorstore.save_local(VECTORSTORE_PATH)

                # ... [rest of your processing code] ...

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
