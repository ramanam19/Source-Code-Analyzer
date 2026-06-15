import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.helper import clone_github_repo, load_repo, split_documents, get_embeddings

# Load environment variables
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "chroma_db"
CLONED_REPO_PATH = "cloned_repo"


# ── Main Indexing Function ────────────────────────────────────────────────────
def ingest_repository(repo_url: str):
    """Full pipeline: clone → load → split → embed → store in ChromaDB."""

    # Always remove old cloned repo and re-clone fresh
    if os.path.exists(CLONED_REPO_PATH):
        shutil.rmtree(CLONED_REPO_PATH)
        print(f"Removed old cloned repo.")

    # Step 1: Clone the repo fresh
    clone_github_repo(repo_url, CLONED_REPO_PATH)

    # Step 2: Debug — list all files found
    from pathlib import Path
    all_files = list(Path(CLONED_REPO_PATH).rglob("*.*"))
    print(f"Total files found in repo: {len(all_files)}")

    py_files = list(Path(CLONED_REPO_PATH).rglob("*.py"))
    print(f"Python files found: {len(py_files)}")
    for f in py_files:
        print(f"  → {f}")

    # Step 3: Load Python files
    documents = load_repo(CLONED_REPO_PATH)

    if not documents:
        ext_counts = {}
        for f in all_files:
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        print(f"File extensions found: {ext_counts}")
        return None

    # Step 4: Split into chunks
    chunks = split_documents(documents)

    # Step 5: Get Gemini embeddings
    embeddings = get_embeddings()

    # Step 6: Store in ChromaDB
    print("Storing chunks in ChromaDB...")
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")
    return vectorstore

# ── Cleanup Function ──────────────────────────────────────────────────────────
def clear_repository():
    """Remove the cloned repository folder to free disk space."""
    if os.path.exists(CLONED_REPO_PATH):
        shutil.rmtree(CLONED_REPO_PATH)
        print(f"Removed cloned repository at {CLONED_REPO_PATH}")
    else:
        print("No cloned repository found.")


# ── Load Existing ChromaDB ────────────────────────────────────────────────────
def load_vectorstore():
    """Load an existing ChromaDB vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )
    return vectorstore