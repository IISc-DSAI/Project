<!-- # SQLite + Streamlit based UI -->



<!-- commands to run  -->

<!-- uvicorn llm_server:app --host 0.0.0.0 --port 8100 -->
<!-- uvicorn llm_vision:app --host 0.0.0.0 --port 8600 --workers 1 -->
<!-- uvicorn lang>



<!-- MCP reqs -->
<!-- pip install \
    requests \
    google-api-python-client \
    python-dotenv \
    rich \
    mcp \
    sseclient-py \
    aiohttp \
    cachetools \
    trafilatura \
    youtube-transcript-api \
    langchain \
    langchain-ollama \
    sympy \
    pydantic \
    fastapi \
    uvicorn \
    httpx -->

<!--
export GOOGLE_CLOUD_PROJECT="vinay-477817"
export GOOGLE_CLOUD_LOCATION="us-central1"        # or "global" depending on your setup
export GOOGLE_GENAI_USE_VERTEXAI="True"
# optionally:
# export GOOGLE_API_KEY="AQ...."   # you can pass api_key to Client() too
-->
<!-- uvicorn final_langgraph:app --host 0.0.0.0 --port 9000 -->


<!-- python manage.py runserver 0.0.0.0:8000 -->


# AURA: Augmented Utilities and Response Agent

**[Kuldeep](https://github.com/Kuldeep1709)**, 
**[Saksham](https://github.com/Saksham-Maitri)**, 
**[Udit](https://github.com/Udit-Shah-37)**, 
**[Vinay](https://github.com/vinaaaaay)**, 
**[Yash](https://github.com/YasKam24)**


  Multimodal AI Assistant for Technical document understanding

  AURA is a multimodal, retrieval-augmented assistant that helps engineers, technicians, and operators extract information from large technical manuals, scanned documents, product catalogs, and online sources. It combines RAG, MCP, LLMs, vision models, and a modular multi-service backend to produce fast, accurate, context-aware responses.

  ---

  ## Key Features

  - **Intelligent Document QA**
    - Upload PDFs, scans, diagrams, and images
    - Instant answers with citations
    - Technical manual–friendly knowledge extraction

  - **Advanced AI Pipeline**
    - Query rewriting
    - Conversation summarization
    - Image understanding
    - RAG (Retrieval-Augmented Generation)
    - MCP (Web Search + YouTube + Forums)
    - Final answer polishing (Claude 3 Sonnet)

  - **Multimodal Support**
    - Accepts images (base64)
    - Generates image descriptions
    - Uses both vision–language and text LLMs

  - **MCP Integration**
    - Google search (via Serper API)
    - YouTube search and transcript summarization
    - Blog/article extraction with Trafilatura
    - Optional Amazon product suggestions

  - **RAG Pipeline**
    - PDF parsing with MinerU
    - Dual knowledge-graph construction
    - Embeddings using BGE Large v1.5
    - Retrieval via LightRAG / RAG-Anything

  - **Full Chat System**
    - Chat creation, renaming, deletion
    - Conversation history retrieval
    - Message storage with metadata
    - Attachments and pipeline execution logs

  - **Streamlit UI**
    - Modern interactive chat interface
    - Image drag-and-drop
    - Chat renaming and history sidebar
    - Sign-in & sign-up

  ---

  ## Architecture Overview

  AURA follows a three-part architecture:

  - Frontend: Streamlit client
  - Django backend: REST API (authentication, persistence, orchestration)
  - LangGraph microservice: FastAPI service handling the AI pipeline

  Flow: Frontend → Django Backend → LangGraph (FastAPI)

  ---

  ## Components

  **Django Backend**

  - Handles authentication and sessions
  - Stores users, chats, messages, attachments, and pipeline logs
  - Retrieves conversation history and calls LangGraph with query + images
  - Persists AI responses and metadata
  - Uses SQLite for zero-maintenance persistence

  **LangGraph (FastAPI)**

  Handles the AI pipeline, including:

  - Query rewriting and conversation summarization
  - Image description (e.g., Qwen2.5-VL)
  - MCP aggregation (web, YouTube, optional Amazon)
  - RAG with MinerU + vector embeddings
  - Final answer synthesis (Claude 3 Sonnet)

  ---

  ## Backend Database

  SQLite schema (high level):

  | Table              | Purpose                                 |
  |-------------------:|:----------------------------------------|
  | `User`              | Stores user accounts                    |
  | `Chat`              | Chat sessions                           |
  | `Message`           | User + AI messages with metadata       |
  | `Attachment`        | Uploaded images and files               |
  | `PipelineExecution` | Full log of intermediate pipeline steps |

  All relations are cascade-safe.

  ---

  ## AI Models Used

  | Component           | Model                      | Purpose                                 |
  |:-------------------:|:--------------------------:|:----------------------------------------|
  | Query rewriting     | Qwen2.5-1.5B               | Rewriting, summarization, title gen     |
  | Image understanding | Qwen2.5-VL-3B              | Image description and vision features   |
  | Embeddings          | BGE Large v1.5             | Vector embeddings for retrieval         |
  | RAG engine          | LightRAG + MinerU          | KG construction and content extraction  |
  | Final answer        | Claude 3 Sonnet (Bedrock)  | Polished response generation            |
  | MCP summarization   | Gemma 3 4B (Ollama)        | Fast inference for MCP prompts          |

  ---

  ## API Endpoints (Django)

  Authentication

  - `POST /auth/signup` — Create a new user
  - `POST /auth/login` — Login and create a session
  - `GET /auth/me` — Check session status

  Chat management

  - `GET /chats` — List user chats
  - `POST /chats` — Create a chat
  - `PATCH /chats/<id>` — Rename a chat
  - `DELETE /chats/<id>` — Delete a chat

  Messages

  - `GET /chats/<id>/messages` — Retrieve messages with pipeline metadata

  Core AI pipeline

  - `POST /chat` — Send query + images + settings to LangGraph. The backend stores the request and the resulting AI response and pipeline execution logs.

  ---

  ## LangGraph API

  ### POST /process

  Inputs (example):

  ```json
  {
    "query": "...",
    "chat_history": "...",
    "images_base_64": [],
    "mcp": 1,
    "rag": 0,
    "yt_summary": 1
  }
  ```

  Typical outputs (high level):

  - `final_response` — The final synthesized assistant answer
  - `google_links` — Links discovered during MCP
  - `youtube_links` — YouTube results
  - `citations` — Source citations from RAG/MCP
  - `youtube_summary` — Summaries of matched YouTube transcripts
  - `intermediate_steps` — Full pipeline trace for debugging/audit

  ---

  ## How to Run (examples)

  Start the Django backend:

  ```bash
  python manage.py runserver 0.0.0.0:8000
  ```

  Start the LangGraph FastAPI microservice (example):

  ```bash
  uvicorn final_langgraph:app --host 0.0.0.0 --port 9000
  ```

  Start the Streamlit frontend (example):

  ```bash
  streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
  ```

  - Start the LLM API server:

  ```bash
  uvicorn llm_server:app --host 0.0.0.0 --port 8100
  ```

  - Start the LLM vision service:

  ```bash
  uvicorn llm_vision:app --host 0.0.0.0 --port 8600 --workers 1
  ```

  ---

  ## Team

  - Kuldeep — JS Backend, React Frontend
  - Saksham — Pipeline orchestration, LLM experiments
  - Udit — Django backend, DB integration
  - Vinay — Full RAG pipeline, orchestration
  - Yash — MCP, debugging

  ---

  ## References

  - RAG-Anything — https://github.com/HKUDS/RAG-Anything
  - MinerU — https://github.com/opendatalab/MinerU
  - LightRAG — https://github.com/HKUDS/LightRAG
  - Internet Archive
  - YouTube API
  - Serper API


