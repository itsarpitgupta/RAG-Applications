# 🚀 Production RAG Applications Suite

A curated repository of enterprise-grade **Retrieval-Augmented Generation (RAG)** systems exploring diverse architectural patterns: from Agentic Workflows to Incremental Delta Ingestion and Scalable Financial Intelligence.

---

## 📂 Projects Overview

| Project | Name | Core Tech Stack | Key Architectural Features |
| :--- | :--- | :--- | :--- |
| **[Project-1](./Project-1)** | **AI-Powered Investor Intelligence Platform** | FastAPI, Azure AI Search, Azure OpenAI, PostgreSQL, Docker, K8s | Financial document parsing, semantic search, automated KPI extraction, relational KPI persistence, and enterprise cloud deployment. |
| **[Project-2](./Project-2)** | **Agentic RAG Document Search System** | LangGraph, LangChain, FAISS, Groq LLM, Streamlit | Multi-format ingestion (Web, PDF, TXT), state-driven agent graph orchestration, query routing, and real-time interactive UI. |
| **[Project-3](./Project-3)** | **Incremental RAG with Scheduled Delta Ingestion** | APScheduler, FAISS (Incremental Append), Grok LLM / Groq, SQLite, Streamlit | Scheduled background batch ingestion, zero-rebuild vector append (`index.add`), SHA-256 deduplication, traceable source citations. |

---

## 🛠️ Repository Structure

```text
RAG-Applications/
├── Project-1/                   # AI-Powered Investor Intelligence Platform
│   ├── config/                  # App configurations
│   ├── ingestion/               # Document ingestion pipelines
│   ├── k8s/                     # Kubernetes manifests
│   ├── llm/                     # Azure OpenAI integration & prompts
│   ├── rag/                     # RAG orchestrator & search logic
│   ├── routes/                  # FastAPI endpoints
│   ├── database/                # Database models & connections
│   ├── main.py / app.py         # Application entrypoints
│   └── README.md                # Project 1 deep-dive documentation
├── Project-2/                   # Agentic RAG Document Search System
│   ├── src/                     # LangGraph nodes, state, & vectorstore
│   ├── data/                    # Sample documents & URL feeds
│   ├── streamlit_app.py         # Streamlit chat interface
│   └── README.md                # Project 2 deep-dive documentation
├── Project-3/                   # Incremental RAG System with Scheduled Ingestion
│   ├── config/                  # Settings and YAML configurations
│   ├── src/                     # Ingestion, pipeline, FAISS vectorstore, scheduler
│   ├── storage/                 # Document storage & FAISS index mounts
│   ├── app.py                   # Streamlit control panel & search UI
│   └── README.md                # Project 3 deep-dive documentation
├── .gitignore                   # Global exclusions (env, venv, caches, db)
└── README.md                    # Main repository landing page
```

---

## ⚡ Quickstart Guide

### General Prerequisites
- **Python 3.11+** or **3.12+**
- Git
- Package managers: `pip` or [`uv`](https://github.com/astral-sh/uv)

### Running a Project
Each project is fully standalone. Navigate to the project directory, configure your environment variables, and run:

#### Project 1: Investor Intelligence Platform
```bash
cd Project-1
cp .env.example .env     # Populate your Azure OpenAI & Azure AI Search keys
pip install -r requirements.txt
python main.py
```

#### Project 2: Agentic RAG Document Search
```bash
cd Project-2
cp .env.example .env     # Populate your GROQ_API_KEY
pip install -r requirements.txt
streamlit run streamlit_app.py
```

#### Project 3: Incremental RAG with Scheduled Ingestion
```bash
cd Project-3
cp .env.example .env     # Populate your GROK_API_KEY / GROQ_API_KEY
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔒 Security Best Practices
- Never commit active `.env` files containing live API keys.
- All projects provide `.env.example` templates with placeholder variables.
- Vector indices and temporary databases in runtime storage directories are git-ignored by default.

---

## 👤 Author
- **Arpit Kumar** ([@itsarpitgupta](https://github.com/itsarpitgupta))
