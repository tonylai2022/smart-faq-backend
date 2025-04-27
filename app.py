from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import os
import fitz  # PyMuPDF
import docx
import json
from dotenv import load_dotenv
from rag import embed_documents, search_docs, load_index, save_index, split_text

# Load env
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

current_chunks = []
uploaded_files = []
load_index()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json["question"]
    retrieved = search_docs(query)
    context = "\n".join(retrieved)
    prompt = (
        f"You are a helpful assistant. Based only on the following information, "
        f"give a concise and clear answer to the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "stream": False
    }

    try:
        res = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        res.raise_for_status()
        result = res.json()
        full_answer = result["choices"][0]["message"]["content"]
        print("\n📝 Full AI Answer:", full_answer)
        return jsonify({"answer": full_answer})
    except Exception as e:
        print("❌ DeepSeek API error:", e)
        return jsonify({"answer": "Sorry, DeepSeek API call failed."})

def extract_text_from_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

@app.route("/upload", methods=["POST"])
def upload_file():
    global current_chunks, uploaded_files
    file = request.files["file"]
    filename = secure_filename(file.filename)
    if filename not in uploaded_files:
        uploaded_files.append(filename)
    save_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    file.save(save_path)

    raw_text = extract_text_from_file(save_path)
    chunks = split_text(raw_text, chunk_size=300, overlap=20)
    current_chunks.extend(chunks)
    embed_documents(chunks)
    save_index()

    return jsonify({"message": "File uploaded and embedded", "chunks": len(chunks)})

@app.route("/files", methods=["GET"])
def list_files():
    return jsonify({"files": uploaded_files})

@app.route("/reset", methods=["POST", "OPTIONS"])
def reset_memory():
    if request.method == "OPTIONS":
        response = make_response('', 204)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

    global current_chunks, uploaded_files
    current_chunks.clear()
    uploaded_files.clear()
    try:
        embed_documents([])
    except Exception as e:
        print("⚠️ Skipped embedding empty list:", e)
    save_index()
    return jsonify({"message": "Memory cleared."})

if __name__ == "__main__":
    app.run(debug=True)
