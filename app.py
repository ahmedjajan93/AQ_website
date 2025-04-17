import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = "my_vector_db"

# Streamlit UI
st.title("🌐 Website QA & Contact Info Extractor (WebBaseLoader Edition)")
url = st.text_input("Enter website URL:", placeholder="https://example.com")
query = st.text_area("Ask a question about the website:", height=200)


if st.button("Submit", type='primary'):
    if not url:
        st.warning("Please enter a website URL.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            try:
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                loader = WebBaseLoader(url)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = url

                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

                vectorstore = FAISS.from_documents(chunks, embedding_model)
                vectorstore.save_local(VECTORSTORE_PATH)

                retriever = vectorstore.as_retriever()

                # Regex contact info
                full_text = "\n".join([doc.page_content for doc in docs])
                regex_contacts = extract_contact_info(full_text)

                st.subheader("📇 Regex Contact Info")
                for key, values in regex_contacts.items():
                    if values:
                        st.markdown(f"**{key.capitalize()}:**")
                        for v in values:
                            st.markdown(f"- {v}")
                    else:
                        st.markdown(f"**{key.capitalize()}:** Not found")

                # Handle user question
                question_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
You are an expert technical assistant.

Use the context below to answer the user's question in a detailed, helpful, and beginner-friendly way.

Context:
{context}

Question:
{question}
"""
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": question_prompt},
                    return_source_documents=False
                )

                result = qa_chain.run(query)
                st.subheader("💬 Answer to Your Question")
                st.markdown(result)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
