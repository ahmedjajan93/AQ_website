from langchain_community.document_loaders import SeleniumURLLoader
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

# Replace Selenium with SeleniumBase
from seleniumbase import Driver

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

VECTORSTORE_PATH = "my_vector_db"

# Streamlit UI
st.title("🔎 Website QA & Contact Info Extractor")

url = st.text_input("Enter website URL:", placeholder="https://example.com")
query = st.text_area("Ask a question about the website:", height=200)

def get_custom_loader(url):
    # Use SeleniumBase Driver which handles Chrome installation automatically
    driver = Driver(browser="chrome", headless=True, uc=True)
    loader = SeleniumURLLoader(urls=[url])
    loader.driver = driver
    return loader

def extract_contact_info(text: str):
    # Extract contact information using regex
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"(?:(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}", text)
    websites = re.findall(r"(https?://[^\s\"\'<>]+)", text)
    addresses = re.findall(
        r"\d{1,5}\s[\w\s]{1,30}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Square|Sq|Court|Ct|Parkway|Pkwy)[^\n,\.]*", 
        text
    )
    return {
        "emails": list(set(emails)),
        "phones": list(set(phones)),
        "websites": list(set(websites)),
        "addresses": list(set(addresses))
    }

if st.button("Submit", type='primary'):
    if not url:
        st.warning("Please enter a website URL.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            # Load or build vectorstore
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            if os.path.exists(VECTORSTORE_PATH):
                vectorstore = FAISS.load_local(
                    VECTORSTORE_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
            else:
                try:
                    loader = get_custom_loader(url)
                    docs = loader.load()
                    loader.driver.quit()

                    for doc in docs:
                        doc.metadata["source"] = url

                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    vectorstore = FAISS.from_documents(chunks, embedding_model)
                    vectorstore.save_local(VECTORSTORE_PATH)
                except Exception as e:
                    st.error(f"Failed to load website: {str(e)}")
                    st.stop()

            retriever = vectorstore.as_retriever()

            # Regex contact info
            full_text = "\n".join([doc.page_content for doc in retriever.get_relevant_documents("contact")])
            regex_contacts = extract_contact_info(full_text)

            st.subheader("📇 Regex Contact Info")
            for key, values in regex_contacts.items():
                if values:
                    st.markdown(f"**{key.capitalize()}:**")
                    for v in values:
                        st.markdown(f"- {v}")
                else:
                    st.markdown(f"**{key.capitalize()}:** Not found")

            # LLM refined contact info
            llm = ChatOpenAI(
                model="google/gemma-3-27b-it:free",
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=1024
            )

            contact_prompt = PromptTemplate(
                input_variables=["context"],
                template="""Extract the following contact details from the text below, if available:
                - Official phone number
                - Email address
                - Physical address
                - Official website URL

                Context:
                {context}

                Return the output in this format:

                Phone: ...
                Email: ...
                Address: ...
                Website: ...
                """
            )

            refiner_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": contact_prompt},
                return_source_documents=False
            )

            refined_contacts = refiner_chain.run("Extract contact information")
            st.subheader("🤖 Refined Contact Info (LLM)")
            st.markdown(refined_contacts)

            # User question
            question_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert technical assistant.

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

            try:
                result = qa_chain.run(query)
                st.subheader("💬 Answer to Your Question")
                st.markdown(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
