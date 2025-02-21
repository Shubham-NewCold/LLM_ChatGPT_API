import os
import pdfplumber
import numpy as np
from flask import Flask, render_template_string, request
from dotenv import load_dotenv
load_dotenv()

import openai

# ---- Local Embedding with Sentence Transformers ----
from sentence_transformers import SentenceTransformer

###################
# CONFIG & GLOBALS
###################
openai.api_key = os.getenv("OPENAI_API_KEY")

# If you have ChatGPT access but not embeddings from OpenAI, that's fine—
# we'll do local embeddings and then pass final queries to ChatGPT for Q&A.

app = Flask(__name__)

PDF_PATH = "input.pdf"
CHUNK_SIZE = 250  # ~250 words
EMBEDDED_CHUNKS = []  # will store (chunk_text, embedding_vector)

# Choose a local model. "multi-qa-MiniLM-L6-cos-v1" is good for Q&A retrieval.
# Another popular one is "all-MiniLM-L6-v2".
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# We'll also specify which ChatCompletion model to use for final answers.
# "gpt-3.5-turbo" or "gpt-4" if you have access.
CHAT_MODEL = "gpt-3.5-turbo"

##########################
# 1. PDF TEXT EXTRACTION #
##########################

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(full_text, chunk_size=250):
    words = full_text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

################################
# 2. LOCAL EMBEDDING & STORAGE #
################################

def load_local_embedding_model():
    """
    Loads a SentenceTransformer model from Hugging Face. 
    This does not require OpenAI's Embedding API.
    """
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_chunks(chunks, model):
    """
    Given a list of text chunks and a SentenceTransformer model,
    returns a list of tuples: (chunk_text, embedding_vector).
    """
    print("Embedding all chunks locally... This may take a bit for large PDFs.")
    # Encode all chunks in batches
    embeddings = model.encode(chunks, batch_size=8, show_progress_bar=True)
    result = []
    for chunk_text, emb in zip(chunks, embeddings):
        result.append((chunk_text, emb))
    return result

##################################
# 3. RETRIEVAL VIA COSINE SIM    #
##################################

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_chunks(user_query, embedded_chunks, embedding_model, top_k=3):
    """
    1. Embed the user_query with the local model.
    2. Compute cosine similarity with each chunk.
    3. Sort & return top_k chunk texts.
    """
    query_embedding = embedding_model.encode([user_query])[0]  # single vector
    scored = []
    for chunk_text, chunk_vec in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk_vec)
        scored.append((score, chunk_text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for (score, text) in scored[:top_k]]
    return top_chunks

#############################
# 4. CALLING OPENAI (GPT)   #
#############################

def ask_chatgpt(context, user_query):
    """
    Sends the combined context + user query to ChatGPT via OpenAI's ChatCompletion.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant with deep knowledge of AI and Big Data "
                    "applications in investments. You have access to the following text. "
                    "Answer the user's question based on the text. If you are unsure, say so."
                )
            },
            {
                "role": "user",
                "content": f"Relevant text:\n{context}\n\nUser Query: {user_query}"
            }
        ]

        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Error calling OpenAI API:", e)
        return "Sorry, I couldn't process your request."

########################
# 5. FLASK APP ROUTES  #
########################

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            answer = "Please enter a valid query."
        else:
            # 1) Retrieve top chunks from local embeddings
            top_chunks = find_similar_chunks(user_query, EMBEDDED_CHUNKS, EMBEDDING_MODEL, top_k=3)
            context = "\n\n".join(top_chunks)
            # 2) Ask ChatGPT
            answer = ask_chatgpt(context, user_query)

        return render_template_string(HTML_PAGE, query=user_query, answer=answer)

    # GET request
    return render_template_string(HTML_PAGE, query="", answer="")

######################
# 6. HTML PAGE       #
######################

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI & Big Data Handbook Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 600px; margin: auto; }
        .query-box { width: 100%; padding: 8px; }
        .submit-button { padding: 8px 16px; }
        .answer { margin-top: 20px; padding: 10px; background: #f0f0f0; }
    </style>
</head>
<body>
<div class="container">
    <h1>AI & Big Data Handbook Chatbot</h1>
    <form method="POST" action="/">
        <label for="query">Ask a question about the handbook:</label><br>
        <textarea id="query" name="query" rows="3" class="query-box">{{ query }}</textarea><br><br>
        <button type="submit" class="submit-button">Ask</button>
    </form>
    {% if answer %}
    <div class="answer">
        <strong>Answer:</strong>
        <p>{{ answer }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

########################
# 7. APP STARTUP       #
########################

# We'll keep a global reference to the local embedding model
EMBEDDING_MODEL = None

def load_and_embed_pdf():
    """
    1) Load a local model (Sentence Transformers).
    2) Extract & chunk PDF text.
    3) Embed chunks & store them in EMBEDDED_CHUNKS.
    """
    global EMBEDDING_MODEL, EMBEDDED_CHUNKS

    # 1. Load local embedding model
    EMBEDDING_MODEL = load_local_embedding_model()

    # 2. Extract & chunk PDF
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text, CHUNK_SIZE)
    print(f"Loaded {len(chunks)} chunks from PDF.")

    # 3. Embed chunks
    EMBEDDED_CHUNKS = embed_chunks(chunks, EMBEDDING_MODEL)
    print(f"Embedded {len(EMBEDDED_CHUNKS)} chunks.")


if __name__ == "__main__":
    load_and_embed_pdf()
    app.run(debug=True, host="127.0.0.1", port=5000)