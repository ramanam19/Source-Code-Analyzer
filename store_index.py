import os
import time
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.helper import clone_github_repo, load_repo, split_documents, get_embeddings, detect_stacks

load_dotenv()

# Force /tmp for Render (writable), fallback to local for development
if os.path.exists("/tmp"):
    CHROMA_DB_PATH   = "/tmp/chroma_db"
    CLONED_REPO_PATH = "/tmp/cloned_repo"
else:
    CHROMA_DB_PATH   = os.path.join(os.getcwd(), "chroma_db")
    CLONED_REPO_PATH = os.path.join(os.getcwd(), "cloned_repo")

print(f"Using CHROMA_DB_PATH: {CHROMA_DB_PATH}")
print(f"Using CLONED_REPO_PATH: {CLONED_REPO_PATH}")


def embed_with_retry(chunks, embeddings, chroma_path, batch_size=50):
    """Store chunks in ChromaDB in small batches with retry on 429."""

    # Clean up old DB
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Make sure directory exists and is writable
    os.makedirs(chroma_path, exist_ok=True)
    print(f"ChromaDB path: {chroma_path} (writable: {os.access(chroma_path, os.W_OK)})")

    vectorstore = None
    total = len(chunks)
    print(f"Embedding {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{-(-total // batch_size)} ({len(batch)} chunks)...")

        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=chroma_path,
                    )
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    wait = 35 * (attempt + 1)
                    print(f"  Rate limit hit. Waiting {wait}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    raise e
        else:
            raise Exception("Embedding failed after 5 retries due to rate limits.")

        if i + batch_size < total:
            time.sleep(3)

    print(f"Successfully stored {total} chunks!")
    return vectorstore


def ingest_repository(repo_url: str):
    clone_github_repo(repo_url, CLONED_REPO_PATH)

    stacks = detect_stacks(CLONED_REPO_PATH)
    if not stacks:
        return None, {}
    print(f"Detected stacks: {stacks}")

    documents = load_repo(CLONED_REPO_PATH)
    if not documents:
        return None, stacks

    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = embed_with_retry(chunks, embeddings, CHROMA_DB_PATH, batch_size=50)

    return vectorstore, stacks


def clear_repository():
    if os.path.exists(CLONED_REPO_PATH):
        shutil.rmtree(CLONED_REPO_PATH)
        print("Cleared cloned repo.")


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )