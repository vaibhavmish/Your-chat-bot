import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment
load_dotenv("chatBot.env")

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)

# Page layout
st.set_page_config(page_title="Your Chat bot", layout="wide")
st.title("This is Your-Chat-Bot")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

def load_vectorstore(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore

if pdf_file is not None:
    vectorstore = load_vectorstore(pdf_file)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # ✅ Fix: Call the method
        memory=memory
    )
else:
    st.warning("Please upload a PDF to get started.")
    st.stop()

# App description
st.markdown("""
This chatbot answers questions based on the **PDF file** using **Groq + LangChain**.  
You can type your question or press the mic button to speak.
""")

# Text input
user_input = st.text_input("Enter your question (or press mic to speak)")
response_placeholder = st.empty()  # ✅ Fix: Add parentheses

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        response_placeholder.write("Thinking...")
        response = qa_chain.invoke({"question": user_input})
        final_answer = response["answer"]
        response_placeholder.write(final_answer)
