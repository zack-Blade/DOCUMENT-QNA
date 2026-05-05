# 📄 Document Q&A

An AI-powered web app that lets you upload any PDF and ask questions about it. Built with Flask, FAISS, and Groq's LLaMA model.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.0+-green) ![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange)

---

## 🚀 What It Does

- Upload a PDF document
- Ask any question about its content
- Get accurate, AI-generated answers pulled directly from the document
- Keeps a chat history of all your questions and answers

---

## 🧠 How It Works (RAG Pipeline)

This app uses a technique called **Retrieval-Augmented Generation (RAG)**:

1. **Extract** — reads text from the uploaded PDF
2. **Chunk** — splits the text into smaller pieces
3. **Embed** — converts chunks into vectors using `sentence-transformers`
4. **Search** — when you ask a question, FAISS finds the most relevant chunks
5. **Answer** — the relevant chunks + your question are sent to Groq's LLaMA model, which generates an accurate answer

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| PDF Reading | pdfplumber |
| Embeddings | sentence-transformers |
| Vector Search | FAISS |
| AI Model | Groq (LLaMA 3.3 70B) |

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/zack-Blade/DOCUMENT-QNA.git
cd DOCUMENT-QNA
```

### 2. Install dependencies

```bash
pip install flask pdfplumber sentence-transformers faiss-cpu numpy groq python-dotenv
```

### 3. Add your Groq API key

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Create the uploads folder

```bash
mkdir uploads
```

### 5. Run the app

```bash
python app.py
```

Open your browser and go to `http://127.0.0.1:5000`

---

## 📸 Usage

1. Click **Choose File** and select a PDF
2. Click **Upload** and wait for it to be processed
3. Type your question in the input box
4. Click **Ask** and get your answer

---

## 📁 Project Structure

```
doc-qa/
├── app.py           # Flask backend + RAG logic
├── templates/
│   └── index.html   # Frontend UI
├── uploads/         # Uploaded PDFs (not pushed to GitHub)
├── .env             # API keys (not pushed to GitHub)
└── .gitignore
```

---

## 🔒 Security Notes

- Never commit your `.env` file
- The `uploads/` folder is excluded from Git
- API keys are loaded via environment variables only

---

## 👤 Author

**Zack Blade**  
GitHub: [@zack-Blade](https://github.com/zack-Blade)
