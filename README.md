## Chat With Your Own Document

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask natural-language questions about its content — with answers grounded in the document, source chunks shown for transparency, and full conversation memory for follow-up questions.

Link: https://chat-with-own-doc-ecxsmzcpalhgmk4fdwpappd.streamlit.app/

Built with **LangChain**, **Google Gemini**, **FAISS**, and **Streamlit**.

## Demonstration

This project was built to showcase an end-to-end RAG pipeline — a core pattern behind most modern "chat with your data" applications:

- Document ingestion and chunking strategy
- Vector embeddings and similarity search
- Retrieval-augmented prompting (grounding an LLM's answers in external data instead of relying on its training knowledge)
- Conversational memory across multi-turn Q&A
- A deployable, user-facing interface (not just a notebook/script)

## How does it work?

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store (FAISS)
                                                              ↓
User Question → Embed Question → Similarity Search → Retrieve Top-K Chunks
                                                              ↓
                                   Chunks + Question + Chat History → LLM → Answer
```

1. **Load** — The uploaded PDF is parsed page-by-page using `PyPDFLoader`.
2. **Split** — Pages are broken into overlapping text chunks using `RecursiveCharacterTextSplitter`, so retrieval can find precise, relevant passages rather than entire pages. (See "Chunk size & overlap" below.)
3. **Embed + Store** — Each chunk is converted into a vector embedding (`GoogleGenerativeAIEmbeddings`, model `gemini-embedding-001`) and stored in an in-memory **FAISS** vector index for fast similarity search.
4. **Retrieve** — When a question is asked, it's embedded the same way, and the top-k most semantically similar chunks are retrieved from the index.
5. **Generate** — The retrieved chunks, the question, and prior conversation history are passed to a Gemini chat model (`ChatGoogleGenerativeAI`) via LangChain's `ConversationalRetrievalChain`, which produces a grounded answer and cites which chunks it used.

## Project Structure:

chat-with-your-own-doc/
├── streamlit_app.py     # UI layer only — file upload, chat interface, sidebar controls
├── rag_pipeline.py       # Backend logic — PDF loading, chunking, embedding, retrieval chain
├── requirements.txt      # Python dependencies
└── README.md             # This file


The backend (`rag_pipeline.py`) is UI-agnostic — it has no Streamlit imports, so the same `build_chain()` and `ask()` functions could be reused in a CLI, a FastAPI backend, or a Jupyter notebook.

## Tech Stack:

| Component | Tool | Why |

| Orchestration | LangChain | Industry-standard framework for chaining retrieval + LLM calls 
| LLM | Google Gemini (`gemini-3.8-flash` or similar) | Free tier available, no credit card required 
| Embeddings | Google `gemini-embedding-001` | Pairs with Gemini, free tier 
| Vector store | FAISS (in-memory) | Fast, lightweight, no external DB needed for a demo-scale project 
| PDF parsing | `pypdf` via `PyPDFLoader` | Reliable page-level text extraction 
| Interface | Streamlit | Fast to build a usable, deployable chat UI 
| Deployment | Streamlit Community Cloud | Free hosting directly from GitHub 


## Setup (local)

```bash
git clone <your-repo-url>
cd chat-with-your-own-doc
pip install -r requirements.txt
```

## Run (local)

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser. In the sidebar:
1. Paste your **Google AI API key** — free, no credit card required, from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Upload a PDF
3. Ask questions in the chat box at the bottom

Your API key is entered only in the browser session — it is never stored in any file or committed to the repo.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, and deploy `streamlit_app.py` as the main file
3. `requirements.txt` is installed automatically
4. `runtime.txt` pins the Python version (3.11) — this avoids compatibility issues with newer Python versions (3.13+) that can break certain LangChain/Pydantic internals
5. On the live app, each user pastes their own API key into the sidebar — no key is stored server-side


## Tuning: chunk size & overlap

These two sliders in the sidebar control how the PDF is split before embedding:

- **Chunk size** — how many characters go into each piece. Smaller chunks (500–800) give more precise retrieval, better for dense/technical documents. Larger chunks (1200–2000) preserve more context, better for narrative or conversational text.
- **Chunk overlap** — how many characters repeat between consecutive chunks, so a sentence or idea isn't lost if it falls on a chunk boundary.

**Default:** 1000 / 150 — a reasonable middle ground for most PDFs. If answers feel vague, try smaller chunks; if answers feel like they're missing surrounding context, increase overlap or chunk size.

Other tunable parameters (in `build_chain()`, `rag_pipeline.py`):
- `top_k` — number of chunks retrieved per question (default: 4). Increase for longer/denser documents.
- `model_name` — which Gemini chat model to use (selectable in the sidebar).


## Possible Extensions

- Support multiple PDFs in a single session (multi-document retrieval)
- Persist the vector index so re-uploading the same PDF doesn't re-embed it
- Add clickable page-number citations instead of plain text excerpts
- Swap FAISS for a hosted vector DB (Pinecone, Weaviate, Qdrant) for production use
- Add evaluation metrics (retrieval precision, answer faithfulness) to benchmark different chunk sizes
