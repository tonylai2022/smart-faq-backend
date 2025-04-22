import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
chunk_data = []
uploaded_files = []

INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"
FILES_PATH = "data/files.pkl"

def embed_documents(texts):
    global chunk_data, index
    if not texts:
        return  # Skip embedding empty input
    vectors = model.encode(texts)
    chunk_data.extend(texts)
    index.add(vectors)
    save_index()
    print("🔍 Embedded Chunks:")
    for i, chunk in enumerate(texts):
        print(f"[{i+1}] {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

def search_docs(query):
    q_vec = model.encode([query])
    D, I = index.search(q_vec, k=1)
    return [chunk_data[i] for i in I[0]] if I[0][0] != -1 else []

def save_index():
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_data, f)
    with open(FILES_PATH, "wb") as f:
        pickle.dump(uploaded_files, f)

def load_index():
    global index, chunk_data, uploaded_files
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunk_data = pickle.load(f)
    if os.path.exists(FILES_PATH):
        with open(FILES_PATH, "rb") as f:
            uploaded_files = pickle.load(f)

def split_text(text, chunk_size=300, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks
