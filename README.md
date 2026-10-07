# RockyBot: News Research Tool 📈 (Gemini Edition)

This is a Retrieval-Augmented Generation (RAG) project adapted from the **Codebasics LangChain News Research Tool** project. Instead of OpenAI, this version uses **Google Gemini** for LLM generation and vector embeddings.

---

## 🌟 Key Features

- **Data Ingestion**: Extract content from web URLs using `UnstructuredURLLoader`.
- **Text Chunking**: Chunk text into manageable segments using `RecursiveCharacterTextSplitter`.
- **Gemini Embeddings**: Convert text chunks into embeddings using `GoogleGenerativeAIEmbeddings` (`models/text-embedding-004`).
- **Vector Database**: Index & store embeddings locally with `FAISS`.
- **Question Answering**: Retrieve context-relevant chunks and query **Gemini** (`gemini-1.5-flash`) via `RetrievalQAWithSourcesChain`.
- **Source Citations**: Display exact article URLs used to answer your question.

---

## 🛠️ Requirements & Installation

### 1. Create & Activate Virtual Environment
```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Configuration

1. Create a `.env` file in the project root folder (or copy `.env.example`):
   ```powershell
   copy .env.example .env
   ```
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).
3. Add your key to `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   *(Or enter your key directly in the Streamlit UI sidebar).*

---

## 🚀 Running the Web App

Run the Streamlit application:
```powershell
streamlit run main.py
```

---

## 🔄 OpenAI vs. Gemini Implementation Comparison

| Feature | Codebasics (Original OpenAI) | Our Implementation (Google Gemini) |
| :--- | :--- | :--- |
| **LLM Model** | `OpenAI()` | `ChatGoogleGenerativeAI(model="gemini-1.5-flash")` |
| **Embedding Model** | `OpenAIEmbeddings()` | `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")` |
| **Vector Index Storage** | Pickled `.pkl` file | `FAISS.save_local()` / `FAISS.load_local()` |
| **LangChain Package** | `langchain` | `langchain-google-genai` |
