# OpenAI API Quickstart Examples

A collection of practical Python scripts demonstrating common use cases with the OpenAI API. Perfect for developers looking for quick, working examples of how to integrate OpenAI's models into their projects.

## 🚀 What's Inside

This repository contains ready-to-use scripts for:

- **📄 PDF Analysis** - Extract and query text from PDF documents using GPT
- **🎥 YouTube Transcript Cleaning** - Transform raw transcripts into readable text
- **💬 Simple Chat Interface** - Interactive REPL for chatting with GPT models
- **🔍 RAG (Retrieval-Augmented Generation)** - Build a question-answering system over your documents

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## 🛠️ Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd openai-quickstart
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```
   
   Or export it as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

## 📚 Scripts Overview

### 1. Simple Chat (`main.py`)

A minimal interactive chat interface using OpenAI's Chat Completions API.

```bash
python main.py
```

**Features:**
- Simple REPL interface
- Configurable model and temperature
- Clean error handling

**Example:**
```python
from main import ask_chat
response = ask_chat("Explain quantum computing in simple terms")
```

---

### 2. PDF Analysis (`analyze_pdf.py`)

Extract text from PDFs and query them using GPT. Handles large documents by chunking them intelligently.

```bash
python analyze_pdf.py path/to/document.pdf "What are the main findings?"
```

**Features:**
- Automatic text extraction from PDFs
- Smart chunking with token limits
- Page number citations
- Configurable system prompts

**Example:**
```bash
python analyze_pdf.py ftc_fy2022_annualreport.pdf "Give me relevant cases for healthcare mergers"
```

---

### 3. YouTube Transcript Cleaning (`youtube_transcripts.py`)

Clean up raw YouTube transcripts (which often lack punctuation and capitalization) into readable, properly formatted text.

```bash
python youtube_transcripts.py raw_transcript.txt
# Outputs: raw_transcript_clean.txt

# Or specify output file:
python youtube_transcripts.py raw_transcript.txt --out cleaned.txt
```

**Features:**
- Restores punctuation and capitalization
- Handles long transcripts with automatic chunking
- Preserves original wording (no content changes)
- Removes timestamps

**Options:**
- `--model`: Choose GPT model (default: `gpt-4o-mini`)
- `--temp`: Sampling temperature (default: `0.2`)

---

### 4. RAG with OpenAI (`run_rag_openai.py`)

Build a Retrieval-Augmented Generation system that answers questions based on your documents.

**Features:**
- Document chunking and embedding
- Vector store using FAISS
- Semantic search and retrieval
- Context-aware responses

**Setup:**
1. Add your document to `my_document.txt`
2. Run the script:
   ```bash
   python run_rag_openai.py
   ```

The script will:
- Load and chunk your document
- Create embeddings using OpenAI's embedding model
- Build a vector store
- Answer questions based on the document content

**Example Questions:**
```python
question = "Where does Dr. Thorne live?"
question = "What is his cat's name?"
```

---

### 5. RAG with Local Ollama (`run_rag.py`)

Alternative RAG implementation using local Ollama models (no API key required).

**Requirements:**
- [Ollama](https://ollama.ai/) installed locally
- `llama3` model downloaded

```bash
ollama pull llama3
python run_rag.py
```

---

## 🗂️ Project Structure

```
openai-quickstart/
├── main.py                 # Simple chat interface
├── analyze_pdf.py          # PDF analysis and querying
├── youtube_transcripts.py  # Transcript cleaning utility
├── run_rag_openai.py      # RAG with OpenAI embeddings
├── run_rag.py             # RAG with local Ollama
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── youtube_transcripts/   # Example transcript files
```

## 💡 Use Cases

These scripts are great starting points for:

- **Document Q&A Systems** - Build chatbots that answer questions about your documents
- **Content Processing** - Clean and format transcripts, summaries, or extracted text
- **Research Tools** - Quickly analyze and query large PDFs or reports
- **Learning** - Understand how to integrate OpenAI's API into your projects

## 🔧 Key Technologies

- **OpenAI Python SDK** - Official Python client for OpenAI API
- **LangChain** - Framework for building LLM applications
- **FAISS** - Vector similarity search (by Facebook AI)
- **pdfplumber** - PDF text extraction
- **tiktoken** - Token counting for GPT models
- **python-dotenv** - Environment variable management

## 📝 Notes

- All scripts use environment variables for API keys (no hardcoded secrets)
- The `.env` file is gitignored for security
- Scripts include error handling and helpful error messages
- Token limits are configured to work with GPT-4o-mini by default

## 🤝 Contributing

Found a bug or have an improvement? Feel free to open an issue or submit a pull request!

## 📄 License

This project is open source and available for learning and modification.

---

**Happy coding! 🎉**

If you find these examples helpful, consider giving this repo a ⭐
