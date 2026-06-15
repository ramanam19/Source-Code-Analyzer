import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from store_index import ingest_repository, load_vectorstore, clear_repository, CHROMA_DB_PATH

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global variables
vectorstore = None
qa_chain = None


# ── Initialize QA Chain ───────────────────────────────────────────────────────
def initialize_qa_chain(vs):
    """Initialize the conversational QA chain with Gemini LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    return chain


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ingest", methods=["POST"])
def ingest():
    """Clone and index a GitHub repository."""
    global vectorstore, qa_chain

    data = request.get_json()
    repo_url = data.get("repo_url", "").strip()

    if not repo_url:
        return jsonify({"error": "Please provide a GitHub URL."}), 400

    try:
        vectorstore = ingest_repository(repo_url)
        if vectorstore is None:
            return jsonify({"error": "No Python files found. Please make sure the repository contains .py files."}), 400
        qa_chain = initialize_qa_chain(vectorstore)
        return jsonify({"message": "Repository ingested successfully! You can now ask questions."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message and return the LLM response."""
    global qa_chain, vectorstore

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if qa_chain is None:
        if os.path.exists(CHROMA_DB_PATH):
            vectorstore = load_vectorstore()
            qa_chain = initialize_qa_chain(vectorstore)
        else:
            return jsonify({"error": "No repository ingested yet. Please provide a GitHub URL first."}), 400

    # Retry up to 3 times on rate limit
    import time
    for attempt in range(3):
        try:
            result = qa_chain.invoke({"question": question})
            answer = result.get("answer", "I could not find an answer.")
            return jsonify({"answer": answer})
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                if attempt < 2:
                    time.sleep(15)  # Wait 15 seconds then retry
                    continue
                else:
                    return jsonify({"answer": "⚠️ Rate limit reached. Please wait 60 seconds and try again."})
            return jsonify({"error": err}), 500

@app.route("/clear", methods=["POST"])
def clear():
    """Clear the cloned repository from disk."""
    global vectorstore, qa_chain
    try:
        clear_repository()
        vectorstore = None
        qa_chain = None
        return jsonify({"message": "Repository cleared successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run App ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)