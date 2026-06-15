# 🔍 Source Code Analyzer

An AI-powered application that allows developers to understand any GitHub codebase through natural language Q&A, built using **Retrieval Augmented Generation (RAG)** with **Google Gemini** and **LangChain**.

---

## 🚀 Live Demo
👉 [Source Code Analyzer](https://source-code-analyzer-u6lp.onrender.com/)

---

## 🧠 How It Works

1. **Paste a GitHub URL** → The app clones the repository
2. **Automatic Indexing** → Python files are extracted, split into chunks, and stored in ChromaDB
3. **Ask Questions** → Gemini LLM answers your questions based on the actual source code

---

## 🏗️ Architecture

GitHub Repo URL
↓
GitPython (Clone)
↓
GenericLoader (Extract .py files)
↓
RecursiveCharacterTextSplitter (Context-aware chunks)
↓
Gemini Embeddings (text-embedding-004)
↓
ChromaDB (Vector Store)
↓
Gemini 2.5 Flash LLM + ConversationSummaryMemory
↓
Flask Web Interface
---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Backend | Flask |
| Frontend | HTML + CSS |
| Deployment | Render |

---

## 📁 Project Structure

source-code-analyzer/

├── src/
│   ├── init.py
│   └── helper.py        # Core RAG functions
├── static/
│   └── style.css        # Frontend styles
├── templates/
│   └── index.html       # Web interface
├── app.py               # Flask backend
├── store_index.py       # ChromaDB indexing
├── requirements.txt     # Dependencies
├── Procfile             # Render deployment
└── .env                 # API keys (not committed)

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/source-code-analyzer.git
cd source-code-analyzer
```

### 2. Create virtual environment with Python 3.11
```bash
py -3.11 -m venv llm_app
llm_app\Scripts\activate  # Windows
source llm_app/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the app
```bash
python app.py
```

Visit `http://localhost:5000`

---

## 🔑 Getting a Gemini API Key

1. Go to 👉 https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and paste it into your `.env` file

---

## 💬 Example Questions

Once a repository is loaded, you can ask:

- `What does this project do?`
- `Explain the main functions in helper.py`
- `How is the database connected?`
- `What frameworks and libraries are used?`
- `Explain the authentication logic`
- `How does the model make predictions?`

---

## ⚠️ Notes

- Only **Python (.py)** files are currently supported
- Free tier Gemini API has a limit of **5 requests per minute**
- Render free tier may **spin down** after inactivity — first load may take 30-60 seconds
- Large repositories may take **2-3 minutes** to index

---

## 🙌 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Gemini](https://ai.google.dev/)
- [ChromaDB](https://www.trychroma.com/)
- [Render](https://render.com/)