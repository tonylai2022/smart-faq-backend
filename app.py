from flask import Flask, request, jsonify, make_response, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import os
import fitz  # PyMuPDF
import docx
import json

from rag import embed_documents, search_docs, load_index, save_index, split_text

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

current_chunks = []
uploaded_files = []
load_index()

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

    def stream_response():
        try:
            with requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "tinyllama", "prompt": prompt, "stream": True},
                stream=True
            ) as res:
                for line in res.iter_lines():
                    if line:
                        try:
                            parsed = json.loads(line.decode("utf-8"))
                            token = parsed.get("response", "")
                            yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n❌ Ollama API error: {str(e)}"

    return Response(stream_response(), mimetype="text/plain")

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
