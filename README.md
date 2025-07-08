# 📄 Your PDF Chatbot 🤖

A smart chatbot powered by **LLaMA 3**, **LangChain**, and **Groq API**, built with **Streamlit** — chat with any PDF using natural language!

---

## 🚀 Features

- 🔍 Ask questions about any uploaded PDF
- 🧠 Uses **LangChain** + **FAISS** for context-aware responses
- ⚡ Powered by **Groq API** and **LLaMA 3** for blazing-fast inference
- 🗃️ Keeps conversation memory across your questions
- 💬 Real-time streaming responses with typing effect
- 🎨 Custom styled chat UI with avatars

---

## 🛠️ Tech Stack

- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API
- LLaMA 3

---

## 📁 Folder Structure

```
your-chat-bot/
│
├── chatbot.py           # Main app code
├── styles.css           # Chat UI styling
├── chatBot.env          # Environment variables (.env)
├── .gitignore           # Git ignore file
└── README.md            # You're here!
```

---

## 🧪 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-chat-bot.git
cd your-chat-bot
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit langchain langchain-community langchain-groq python-dotenv sentence-transformers faiss-cpu
```

### 4. Add your Groq API key

Create a file named `chatBot.env` in the root and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

✅ Don’t forget to add `.env` to your `.gitignore`!

---

## ▶️ Run the chatbot

```bash
streamlit run chatbot.py
```

---

## 📌 Usage

1. Upload any **PDF file**
2. Ask questions like:
   - “What is this document about?”
   - “Summarize page 2.”
   - “Who is the author?”
3. Chatbot will stream its response live using LLaMA 3.

---

## 🛡️ Security Note

- Your `.env` file must **not be committed to GitHub**.
- If accidentally exposed, **regenerate your Groq API key immediately**.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.

---

## 📄 License

MIT License. Use it freely, just don’t forget to give credit.