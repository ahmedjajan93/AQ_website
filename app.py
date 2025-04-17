from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Streamlit UI
st.title("🔎 Ask Questions About a Website")

url = st.text_input("Enter website URL:", placeholder="https://example.com")
query = st.text_area("Ask a question about the website:", height=200)

if st.button("Submit",type='primary') and query:
    with st.spinner("Processing..."):

        # Load and parse website
        loader = SeleniumURLLoader(urls=[url])
        docs = loader.load()

        # Split content
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Embed and store in FAISS
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("my_vector_db")

        # Reload vector store
        vectorstore = FAISS.load_local("my_vector_db", embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        # OpenRouter LLM
        llm = ChatOpenAI(
            model="google/gemma-3-27b-it:free",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1024
        )

        # Custom prompt for explanation of code snippets
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert technical assistant.

Use the context below to answer the user's question in a detailed, helpful, and beginner-friendly way.

Highlight any code examples using Markdown and explain each clearly.

Context:
{context}

Question:
{question}
"""
        )

        # Retrieval QA Chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

        # Run QA
        try:
            result = qa_chain.run(query)
            st.write("### Answer")
            st.markdown(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
