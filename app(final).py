import os
import re
import pdfplumber
import numpy as np
import markdown
from flask import Flask, render_template_string, request
from dotenv import load_dotenv
load_dotenv()

import openai
from sentence_transformers import SentenceTransformer

###################
# CONFIG & GLOBALS
###################
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

PDF_PATH = "input.pdf"
EMBEDDED_CLAUSES = []  # will store (clause_title, clause_text, embedding_vector)
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# We'll also specify which ChatCompletion model to use for final answers.
CHAT_MODEL = "gpt-3.5-turbo"

###############################
# 1. PDF TEXT EXTRACTION      #
###############################
def extract_text_from_pdf(pdf_path):
    """
    Reads all pages from the PDF and returns a single string.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

###########################
# 2. CLAUSE-BASED CHUNKING
###########################
def chunk_text_by_clause(full_text):
    """
    Splits the PDF text by headings such as 'Clause 1', 'Clause 2', 'Schedule 3', or a leading number.
    Returns a list of (clause_title, clause_text).
    """
    # Regex to match lines that look like "Clause 9", "Schedule 3", "1. Definitions", etc.
    # Adjust this pattern to fit your PDF's structure more precisely.
    clause_pattern = re.compile(r'^\s*(Clause\s+\d+|Schedule\s+\d+|\d+\.\s+.*)', re.IGNORECASE)

    lines = full_text.split('\n')
    chunks = []
    current_title = "Intro/Preamble"
    current_lines = []

    for line in lines:
        # If this line looks like a clause heading
        if clause_pattern.match(line.strip()):
            # If we already have some lines collected, store them as a chunk
            if current_lines:
                combined_text = "\n".join(current_lines).strip()
                chunks.append((current_title, combined_text))
                current_lines = []
            current_title = line.strip()
        else:
            current_lines.append(line)

    # Add the last chunk
    if current_lines:
        combined_text = "\n".join(current_lines).strip()
        chunks.append((current_title, combined_text))

    return chunks

################################
# 3. LOCAL EMBEDDING & STORAGE #
################################
def load_local_embedding_model():
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_clauses_by_title(clause_pairs, model):
    """
    Given a list of (clause_title, clause_text) and a SentenceTransformer model,
    returns a list of (clause_title, clause_text, embedding_vector).
    """
    print("Embedding clauses locally... This may take a bit for large PDFs.")
    embedded_data = []
    titles = []
    texts = []

    # We'll embed each chunk as 'title + text' to capture context
    for title, text in clause_pairs:
        combined = f"{title}\n{text}"
        titles.append(title)
        texts.append(text)

    # Actually embed all combined strings in one batch
    combined_strs = [f"{t}\n{x}" for (t,x) in clause_pairs]
    embeddings = model.encode(combined_strs, batch_size=8, show_progress_bar=True)

    for (title, text), vector in zip(clause_pairs, embeddings):
        embedded_data.append((title, text, vector))

    return embedded_data

##################################
# 4. RETRIEVAL VIA COSINE SIM    #
##################################
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_clauses(user_query, embedded_clauses, embedding_model, top_k=3):
    """
    1. Embed the user_query with the local model.
    2. Compute cosine similarity with each clause vector.
    3. Sort & return top_k as (clause_title, clause_text).
    """
    query_embedding = embedding_model.encode([user_query])[0]
    scored = []
    for (title, text, vector) in embedded_clauses:
        score = cosine_similarity(query_embedding, vector)
        scored.append((score, title, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:top_k]
    # Return as a list of (title, text)
    result = [(t, c) for (_, t, c) in top_matches]
    return result

#############################
# 5. CALLING OPENAI (GPT)   #
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
                    "You are a legal/contractual Q&A assistant. You have access to the following text. "\
                    "Answer the user's question based ONLY on this text. If you're unsure, say so. "\
                    "Provide references to relevant clauses or schedules ONLY if they explicitly appear in the text. "\
                    "Use the following formatting guidelines:\n"\
                    "## Start headings with '##'\n"\
                    "- Use a bulleted list with dashes ('-') for subpoints\n"\
                    "- Bold all clause references, e.g., **(Clause X.Y)**\n"\
                    "- Include an extra line break after each bullet group.\n"\
                    "- Italicize any disclaimers or notes.\n"\
                    "Use actual HTML tags for formatting:\n"
                    "- <strong> for bold\n"
                    "- <em> for italics\n"
                    "- <u> for underline\n"
                    "If HTML rendering is supported, you may use <u>...</u> for underlining specific text.\n"
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

###################################
# 6. FLASK APP ROUTES & TEMPLATES #
###################################
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lactalis Warehousing Agreement Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 600px; margin: auto; }
        .query-box { width: 100%; padding: 8px; }
        .submit-button { padding: 8px 16px; }
        .answer { margin-top: 20px; padding: 10px; background: #f0f0f0; white-space: pre-wrap; }
    </style>
</head>
<body>
<div class="container">
    <h1>Lactalis Warehousing Agreement Chatbot</h1>
    <form method="POST" action="/">
        <label for="query">Ask a question about the warehousing agreement:</label><br>
        <textarea id="query" name="query" rows="3" class="query-box">{{ query }}</textarea><br><br>
        <button type="submit" class="submit-button">Ask</button>
    </form>
    {% if answer %}
    <div class="answer">
        <strong>Answer:</strong>
        <p>{{ answer|safe }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            answer = "Please enter a valid query."
        else:
            # 1) Retrieve top clauses from local embeddings
            top_clauses = find_similar_clauses(user_query, EMBEDDED_CLAUSES, EMBEDDING_MODEL, top_k=3)

            # Label each clause with its title
            # e.g. '***Clause 1***\nFull text...'
            context_parts = []
            for (title, text) in top_clauses:
                context_parts.append(f"*** {title} ***\n{text}")
            final_context = "\n\n".join(context_parts)

            # 2) Ask ChatGPT and convert the markdown output to HTML
            raw_answer = ask_chatgpt(final_context, user_query)
            answer = markdown.markdown(raw_answer)

        return render_template_string(HTML_PAGE, query=user_query, answer=answer)

    # GET request
    return render_template_string(HTML_PAGE, query="", answer="")

########################
# 7. APP STARTUP       #
########################
EMBEDDING_MODEL = None
EMBEDDED_CLAUSES = []

def load_and_embed_pdf():
    global EMBEDDING_MODEL, EMBEDDED_CLAUSES

    # 1. Load local embedding model
    EMBEDDING_MODEL = load_local_embedding_model()

    # 2. Extract text from PDF
    raw_text = extract_text_from_pdf(PDF_PATH)

    # 3. Clause-based chunking
    clause_pairs = chunk_text_by_clause(raw_text)
    print(f"Found {len(clause_pairs)} clause-level chunks from PDF.")

    # 4. Embed each clause
    EMBEDDED_CLAUSES = embed_clauses_by_title(clause_pairs, EMBEDDING_MODEL)
    print(f"Embedded {len(EMBEDDED_CLAUSES)} clauses.")

if __name__ == "__main__":
    load_and_embed_pdf()
    app.run(debug=True, host="127.0.0.1", port=5000)