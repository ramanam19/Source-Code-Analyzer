import os
from pathlib import Path
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Supported stacks & their file extensions ─────────────────────────────────
STACK_CONFIG = {
    "python": {
        "extensions": [".py"],
        "language":   Language.PYTHON,
        "label":      "Python",
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".ts", ".tsx", ".mjs"],
        "language":   Language.JS,
        "label":      "JavaScript / TypeScript",
    },
    "java": {
        "extensions": [".java"],
        "language":   Language.JAVA,
        "label":      "Java",
    },
    "html": {
        "extensions": [".html", ".htm"],
        "language":   Language.HTML,
        "label":      "HTML",
    },
    "css": {
        "extensions": [".css", ".scss", ".sass"],
        "language":   None,
        "label":      "CSS / SCSS",
    },
    "config": {
        "extensions": [".json", ".env.example", ".yaml", ".yml", ".xml", ".gradle", ".properties"],
        "language":   None,
        "label":      "Config Files",
    },
}

# Extensions that need plain text loading (no LanguageParser support)
PLAIN_TEXT_EXTENSIONS = {".css", ".scss", ".sass", ".json", ".yaml",
                          ".yml", ".xml", ".gradle", ".properties", ".env.example"}

# All supported extensions flat list
ALL_EXTENSIONS = [ext for cfg in STACK_CONFIG.values() for ext in cfg["extensions"]]


# ── 1. Clone GitHub Repository ───────────────────────────────────────────────
def clone_github_repo(repo_url: str, clone_path: str = "cloned_repo"):
    if os.path.exists(clone_path):
        import shutil
        shutil.rmtree(clone_path)
        print(f"Removed old cloned repo.")
    print(f"Cloning {repo_url} ...")
    Repo.clone_from(repo_url, clone_path)
    print("Cloning complete!")
    return clone_path


# ── 2. Detect which stacks are present ───────────────────────────────────────
def detect_stacks(repo_path: str) -> dict:
    """Scan repo and return which stacks/languages are found."""
    found = {}
    all_files = list(Path(repo_path).rglob("*"))

    for stack, cfg in STACK_CONFIG.items():
        matched = [f for f in all_files if f.suffix.lower() in cfg["extensions"]]
        if matched:
            found[stack] = {"label": cfg["label"], "count": len(matched)}

    return found


# ── 3. Load files using LanguageParser where possible ────────────────────────
def load_repo(repo_path: str):
    documents = []
    repo = Path(repo_path)

    # Skip folders that aren't useful
    SKIP_DIRS = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "env", "dist", "build", ".next", "target", ".gradle",
        "bin", "obj", ".idea", ".vscode", "coverage", ".nyc_output"
    }

    def is_skipped(path: Path) -> bool:
        return any(part in SKIP_DIRS for part in path.parts)

    print(f"Loading source files from {repo_path}...")

    # Group extensions by language for LanguageParser
    for stack, cfg in STACK_CONFIG.items():
        lang = cfg["language"]
        exts = [e for e in cfg["extensions"] if e not in PLAIN_TEXT_EXTENSIONS]

        if not lang or not exts:
            continue

        matched_files = [
            f for f in repo.rglob("*")
            if f.suffix.lower() in exts and not is_skipped(f)
        ]

        if not matched_files:
            continue

        try:
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob="**/*",
                suffixes=exts,
                parser=LanguageParser(language=lang, parser_threshold=100),
                exclude=list(SKIP_DIRS),
            )
            docs = loader.load()
            documents.extend(docs)
            print(f"  [{cfg['label']}] Loaded {len(docs)} document(s)")
        except Exception as e:
            print(f"  [{cfg['label']}] Parser failed: {e}, falling back to plain text")
            # Fallback: plain text load
            for f in matched_files:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": str(f), "language": stack}
                        ))
                except Exception:
                    pass

    # Plain text extensions (CSS, JSON, YAML, etc.)
    for f in repo.rglob("*"):
        if f.suffix.lower() in PLAIN_TEXT_EXTENSIONS and not is_skipped(f):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(f), "language": "config/style"}
                    ))
            except Exception:
                pass

    print(f"Total documents loaded: {len(documents)}")
    return documents


# ── 4. Split documents ────────────────────────────────────────────────────────
def split_documents(documents):
    print("Splitting documents into chunks...")
    chunks = []

    # Group by language for language-specific splitting
    lang_map = {
        "python":     Language.PYTHON,
        "javascript": Language.JS,
        "java":       Language.JAVA,
        "html":       Language.HTML,
    }

    for lang_key, lang_enum in lang_map.items():
        lang_docs = [
            d for d in documents
            if lang_key in d.metadata.get("language", "").lower()
            or lang_key in d.metadata.get("source", "").lower()
        ]
        if lang_docs:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum, chunk_size=1500, chunk_overlap=200
            )
            chunks.extend(splitter.split_documents(lang_docs))

    # Remaining documents (config, css, etc.)
    handled_sources = {d.metadata.get("source") for d in documents
                       if any(k in d.metadata.get("language","").lower()
                              or k in d.metadata.get("source","").lower()
                              for k in lang_map)}
    remaining = [d for d in documents if d.metadata.get("source") not in handled_sources]

    if remaining:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks.extend(splitter.split_documents(remaining))

    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ── 5. Gemini Embeddings ──────────────────────────────────────────────────────
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )