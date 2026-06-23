import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.helper import clone_github_repo, load_repo, split_documents, get_embeddings, detect_stacks

load_dotenv()

CHROMA_DB_PATH  = "/tmp/chroma_db"
CLONED_REPO_PATH = "/tmp/cloned_repo"


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

    # Step 4: Split
    chunks = split_documents(documents)

    # Step 5: Embed & store
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB!")
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