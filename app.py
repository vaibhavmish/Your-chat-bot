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
import time

# Load environment variables (API keys, etc.)
load_dotenv("chatBot.env")

# Initialize LLM with streaming enabled
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    streaming=True,
)

# Set page config
st.set_page_config(page_title="Your Chat bot", layout="wide")
st.title("📄 Your PDF Chatbot 🤖")

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style", unsafe_allow_html=True)

# Upload PDF file
pdf_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

# Function to load and embed PDF content
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

# Process uploaded file
if pdf_file is not None:
    vectorstore = load_vectorstore(pdf_file)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
else:
    st.warning("Please upload a PDF to get started.")
    st.stop()

# App introduction
st.markdown("""
Welcome! This chatbot answers questions based on the **PDF you uploaded** using **LangChain + Groq + LLaMA 3**.  
Just type your question below to start chatting!
""")

# Show uploaded filename
st.write(f"✅ Loaded file: `{pdf_file.name}`")

# Text input
user_input = st.chat_input("💬 Ask something about your PDF:")

# Response placeholder
response_placeholder = st.empty()
final_answer = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]


# Streaming and typing effect
if user_input and user_input.strip() != "":
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    response_placeholder.markdown("<div class ='user-bubble'>🤔 Thinking...</div>",unsafe_allow_html=True)

    for chunk in qa_chain.stream(
        {"question": user_input},
        config={"configurable": {"session_id": "user1"}}
    ):
        token = chunk.get("answer", "")
        final_answer += token
        response_placeholder.markdown(f"<div class='bot-bubble'>{final_answer}▌</div>", unsafe_allow_html=True)
        time.sleep(0.02)
        
    # Final answer without cursor
    response_placeholder.markdown(f"<div class='bot-bubble'>{final_answer}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "bot", "message": final_answer})


for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(
            f"""
            <div class="chat user">
                <img src="user_icon.png" class="avatar">
                <div class="bubble user-bubble">{chat["message"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat bot">
                <img src="bot_icon.png" class="avatar">
                <div class="bubble bot-bubble">{chat["message"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
