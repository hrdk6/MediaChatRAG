# MediaChatRAG

MediaChatRAG is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to interact with PDFs and YouTube videos using natural language questions.

# Features

* Chat with PDF documents
* Chat with YouTube video transcripts
* Semantic search using vector embeddings
* Context-aware AI responses
* Clean Streamlit UI

# Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Google Gemini API

# How It Works

1. Upload a PDF or provide a YouTube link
2. Content is converted into embeddings
3. Embeddings are stored in a FAISS vector database
4. User asks questions
5. Relevant context is retrieved
6. Gemini generates the final answer

## Installation

```bash
pip install -r requirements.txt
```

# Run Locally

```bash
streamlit run app.py
```

# Environment Variables

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_api_key
```

# Author

Hardik Gaonkar
