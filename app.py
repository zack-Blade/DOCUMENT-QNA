from flask import Flask, request, jsonify, render_template
import pdfplumber
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chunks = []
vectorizer = None
chunk_vectors = None

def extract_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def split_chunks(text, size=500):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chunks, vectorizer, chunk_vectors
    file = request.files['pdf']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text(path)
    chunks = split_chunks(text)

    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)

    return jsonify({"message": f"PDF processed. {len(chunks)} chunks indexed."})

@app.route('/ask', methods=['POST'])
def ask():
    global chunks, vectorizer, chunk_vectors
    question = request.json.get('question')

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, chunk_vectors).flatten()
    top_indices = scores.argsort()[-3:][::-1]
    relevant = [chunks[i] for i in top_indices]
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