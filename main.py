import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import requests
from langchain_ollama import OllamaEmbeddings
import tempfile
from pinecone import Pinecone, ServerlessSpec
from sympy.codegen import Print
import json
# ============== SETTINGS ==============
PINECONE_API_KEY = "pcsk_2sWnqo_FeFsrzJgnecu4W1mKyjdK4nzJgFJgemujA8RGUAU48dWRtgiWm7iNmsSx5Zi8kQ"
PINECONE_ENV = "aws-us-east-1"  # like "gcp-starter"
INDEX_NAME = "testdb384"

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "medllama2:latest"  # or whatever model you have locally
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)  # :contentReference[oaicite:10]{index=10}

# ---------- Create Index if Needed ----------
if not pc.has_index(INDEX_NAME):  # :contentReference[oaicite:11]{index=11}
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # :contentReference[oaicite:12]{index=12}
    )

# ---------- Prepare Embedding Model ----------
embed_model = OllamaEmbeddings(model="all-minilm")

# ---------- Streamlit UI ----------
st.title("📚 RAG App with Streamlit + Pinecone + Ollama")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
text_splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and chunk
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    text   = "\n".join([p.page_content for p in pages])
    chunks = text_splitter.split_text(text)
    st.write(f"Split into **{len(chunks)} chunks**.")

    # Embed & upsert
    index   = pc.Index(INDEX_NAME)  # :contentReference[oaicite:13]{index=13}
    vectors = embed_model.embed_documents(chunks)
    ids     = [f"doc-{i}" for i in range(len(vectors))]
    meta    = [{"text": c} for c in chunks]
    index.upsert(vectors=list(zip(ids, vectors, meta)))

    st.success("✅ Uploaded to Pinecone!")
else:
    st.error("Embedding returned empty vectors. Check your model or input.")

# ===== User Query Part =====
query = st.text_input("Ask a question about your PDF:")

if query:
    st.write("Thinking...")

    # Search Pinecone
    index = pc.Index(INDEX_NAME)

    # 4. Now you can query safely
    query_vector = embed_model.embed_query(query)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    print(results ,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444")
    context = "\n\n".join([match['metadata']['text'] for match in results['matches']])

    # Query Ollama
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant. Convert the information into a human readable answer."},
            {"role": "user", "content": f"Use this context to answer:\n\n{context}\n\nQuestion: {query}"}
        ]
    }

    response = requests.post(OLLAMA_ENDPOINT, json=payload,stream=False)
    full_answer = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        # each line is a JSON object like {"message": {"role": "...", "content": "..."}}
        try:
            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content", "")
            full_answer += content
        except json.JSONDecodeError:
            # skip lines that aren’t valid JSON
            continue

    # now `full_answer` is your human-readable answer in one piece

    if response.status_code == 200:
        st.success(full_answer)
    else:
        st.error("Failed to get response from Ollama.")

