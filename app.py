from flask import Flask, request, jsonify, render_template
import pdfplumber
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = []
index = None

def extract_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def split_chunks(text, size=500):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index(chunk_list):
    embeddings = model.encode(chunk_list)
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chunks, index
    file = request.files['pdf']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text(path)
    chunks = split_chunks(text)
    index = build_index(chunks)

    return jsonify({"message": f"PDF processed. {len(chunks)} chunks indexed."})

@app.route('/ask', methods=['POST'])
def ask():
    global chunks, index
    question = request.json.get('question')

    query_vec = model.encode([question])
    query_vec = np.array(query_vec).astype('float32')

    _, indices = index.search(query_vec, k=3)
    relevant = [chunks[i] for i in indices[0]]
    context = '\n\n'.join(relevant)

    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({"answer": response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)