import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "static/style.css",
    "templates/index.html",
    "app.py",
    "store_index.py",
    "requirements.txt",
    ".env",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        print(f"Creating empty file: {filepath}")
    else:
        print(f"File already exists: {filepath}")