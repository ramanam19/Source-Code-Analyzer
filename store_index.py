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


def embed_with_retry(chunks, embeddings, chroma_path, batch_size=10):
    """Store chunks in ChromaDB in small batches with retry on 429."""

    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    os.makedirs(chroma_path, exist_ok=True)
    print(f"ChromaDB path: {chroma_path} (writable: {os.access(chroma_path, os.W_OK)})")

    vectorstore = None
    total       = len(chunks)
    num_batches = -(-total // batch_size)  # ceiling division
    print(f"Embedding {total} chunks in {num_batches} batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch      = chunks[i : i + batch_size]
        batch_num  = i // batch_size + 1
        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} chunks)...")

        for attempt in range(6):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=chroma_path,
                    )
                else:
                    vectorstore.add_documents(batch)
                break  # success — exit retry loop

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    # Extract retry delay from error message if available
                    wait = 65  # safe default just over 1 minute
                    try:
                        import re
                        match = re.search(r'retry in (\d+)', err)
                        if match:
                            wait = int(match.group(1)) + 5
                    except Exception:
                        pass
                    print(f"  Rate limit hit. Waiting {wait}s (attempt {attempt+1}/6)...")
                    time.sleep(wait)
                else:
                    raise e
        else:
            raise Exception(f"Batch {batch_num} failed after 6 retries due to rate limits.")

        # Small pause between every batch to stay under 100/min limit
        time.sleep(2)

    print(f"Successfully stored {total} chunks in ChromaDB!")
    return vectorstore


def ingest_repository(repo_url: str):
    # Step 1: Clone fresh
    clone_github_repo(repo_url, CLONED_REPO_PATH)

    # Step 2: Detect stacks
    stacks = detect_stacks(CLONED_REPO_PATH)
    if not stacks:
        return None, {}
    print(f"Detected stacks: {stacks}")

    # Step 3: Load files
    documents = load_repo(CLONED_REPO_PATH)
    if not documents:
        return None, stacks

    # Step 4: Split into chunks
    chunks = split_documents(documents)

    # Step 5: Limit chunks for free tier
    # Free tier: 100 embeds/min. With retries this handles ~300 chunks safely.
    MAX_CHUNKS = 300
    if len(chunks) > MAX_CHUNKS:
        print(f"Limiting chunks from {len(chunks)} to {MAX_CHUNKS} for free tier.")
        chunks = chunks[:MAX_CHUNKS]

    # Step 6: Embed & store
    embeddings  = get_embeddings()
    vectorstore = embed_with_retry(chunks, embeddings, CHROMA_DB_PATH, batch_size=10)

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