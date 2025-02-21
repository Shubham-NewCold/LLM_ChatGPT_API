import os
import openai
from dotenv import load_dotenv
load_dotenv()
import pdfplumber
from flask import Flask, render_template_string, request

#######################
# 1. CONFIG & GLOBALS #
#######################

# If you prefer to use an environment variable for security:
#   export OPENAI_API_KEY=sk-XXXX
# Otherwise, you can hardcode your key:
# openai.api_key = "YOUR_OPENAI_API_KEY_HERE"
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# We'll load and chunk the PDF text once when the server starts.
CONTRACT_CHUNKS = []
CHUNK_SIZE = 250  # number of words per chunk (tweak as needed)

##########################
# 2. PDF TEXT EXTRACTION #
##########################

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using pdfplumber and returns it as a single string.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=250):
    """
    Splits text into chunks of chunk_size words each. Returns a list of chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # If there's a remainder
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

################################
# 3. SIMPLE NAIVE RETRIEVAL    #
################################

def find_relevant_chunks(chunks, query, top_k=3):
    """
    Naive approach: Ranks chunks by how many times the query appears in them.
    Returns the top_k chunks that have the highest count of query words.
    """
    query_lower = query.lower()
    scored_chunks = []

    for chunk in chunks:
        # Count occurrences of each word from query in the chunk
        # For simplicity, do substring matching:
        count = chunk.lower().count(query_lower)
        scored_chunks.append((count, chunk))

    # Sort chunks by the count in descending order
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return the text portions of the top_k scored chunks
    best_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    # If nothing has a score > 0, we'll just return the top chunk (or empty)
    if not best_chunks:
        return [scored_chunks[0][1]] if scored_chunks else []

    return best_chunks

#############################
# 4. CALLING OPENAI (GPT-3.5)
#############################

def ask_chatgpt(context, user_query):
    """
    Sends the combined context and user query to ChatGPT via OpenAI's API.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. You have access to the following contract text. "
                    "Answer the question based ONLY on this text, or say you are not sure if the "
                    "answer isn't there."
                )
            },
            {
                "role": "user",
                "content": f"Contract text:\n{context}\n\nQuestion: {user_query}"
            }
        ]

        response = openai.chat.completions.create(
            #api_key="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",
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print("Error calling OpenAI API:", e)
        return "Sorry, I couldn't process your request."

########################
# 5. FLASK APP ROUTES  #
########################

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Displays a simple form. When the user submits a query, we call the naive retrieval,
    then send the top chunks + user query to OpenAI for an answer.
    """
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            answer = "Please enter a valid query."
        else:
            # Retrieve relevant chunks
            relevant_chunks = find_relevant_chunks(CONTRACT_CHUNKS, user_query)
            context = "\n\n".join(relevant_chunks)

            # Call ChatGPT
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
    <title>Contract Chatbot</title>
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
    <h1>Contract Chatbot</h1>
    <form method="POST" action="/">
        <label for="query">Ask a question about the contract:</label><br>
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

######################
# 7. APP STARTUP     #
######################

def load_pdf_into_memory():
    """
    Extract text from PDF and chunk it. Store globally in CONTRACT_CHUNKS.
    """
    global CONTRACT_CHUNKS
    pdf_text = extract_text_from_pdf("input.pdf")
    CONTRACT_CHUNKS = chunk_text(pdf_text, CHUNK_SIZE)
    print(f"Loaded PDF with {len(CONTRACT_CHUNKS)} chunks.")

if __name__ == "__main__":
    load_pdf_into_memory()
    app.run(debug=True, host="127.0.0.1", port=5000)