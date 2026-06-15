import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ── 1. Clone GitHub Repository ──────────────────────────────────────────────
def clone_github_repo(repo_url: str, clone_path: str = "cloned_repo"):
    """Clone a GitHub repository to a local directory."""
    if os.path.exists(clone_path):
        print(f"Repository already exists at {clone_path}, skipping clone.")
    else:
        print(f"Cloning repository from {repo_url}...")
        Repo.clone_from(repo_url, clone_path)
        print("Cloning complete!")
    return clone_path


# ── 2. Load Python Source Files ──────────────────────────────────────────────
def load_repo(repo_path: str):
    """Load all Python files from the cloned repository."""
    print(f"Loading Python files from {repo_path}...")
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} Python file(s).")
    return documents


# ── 3. Split Code into Chunks ─────────────────────────────────────────────────
def split_documents(documents):
    """Split documents into chunks using context-aware splitting."""
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")
    return chunks


# ── 4. Initialize Gemini Embeddings ──────────────────────────────────────────
def get_embeddings():
    """Initialize and return Google Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    return embeddings