<div align="center">

# 🌾 Project Kisan

### AI-Powered Agricultural Assistant

A hybrid agricultural AI system combining **domain-adapted LLMs, QLoRA, RAG, neural reranking, live web search, and LLM-based tool routing** to provide practical and evidence-aware agricultural assistance.

<p>
  <a href="https://hamdantariq26.github.io/projects/project-kisan">
    <img src="https://img.shields.io/badge/Portfolio-Project%20Kisan-111827?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio">
  </a>
  <img src="https://img.shields.io/badge/Status-Research%20Demo-2563EB?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Year-2026-6B7280?style=for-the-badge" alt="Year">
  <img src="https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge" alt="License">
</p>

<p>
  <img src="https://img.shields.io/badge/Llama%203.2%203B-QLoRA%20Fine--Tuned-8B5CF6?style=flat-square" alt="Llama 3.2 3B">
  <img src="https://img.shields.io/badge/Llama%203.1%208B-Inference-8B5CF6?style=flat-square" alt="Llama 3.1 8B">
  <img src="https://img.shields.io/badge/QLoRA-4--bit%20NF4-F97316?style=flat-square" alt="QLoRA">
  <img src="https://img.shields.io/badge/RAG-FAISS-0EA5E9?style=flat-square" alt="RAG">
  <img src="https://img.shields.io/badge/Reranker-BGE-14B8A6?style=flat-square" alt="BGE">
  <img src="https://img.shields.io/badge/FastAPI-SSE-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Gradio-UI-F97316?style=flat-square&logo=gradio&logoColor=white" alt="Gradio">
</p>

</div>

---

## ✨ What is Project Kisan?

Project Kisan is an **AI-powered agricultural assistant** designed around a simple idea: not every question should be answered in the same way.

Instead of sending every query directly to an LLM, the system uses a dedicated **LLM routing agent** to choose between three response paths:

```text
                         USER QUERY
                              │
                              ▼
                  ┌──────────────────────┐
                  │    LLM ROUTER AGENT  │
                  │   JSON Tool Calling  │
                  └──────────┬───────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
            DIRECT        RAG SEARCH     WEB SEARCH
               │             │             │
               │             ▼             ▼
               │           FAISS         DDGS
               │             │             │
               │        BGE Reranker  BeautifulSoup
               │             │             │
               │          Top 5          Top 10
               │             │             │
               └─────────────┼─────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   RESPONSE MODEL     │
                  │                      │
                  │ Llama 3.2 3B + LoRA  │
                  │          or          │
                  │ Llama 3.1 8B        │
                  └──────────┬───────────┘
                             │
                       Token Streaming
                             │
                             ▼
                        FastAPI SSE
                             │
                             ▼
                         Gradio UI
```

This hybrid architecture is intended to reduce unsupported guessing, improve source relevance, and provide a practical way to combine **parametric knowledge with retrieved evidence**. The system does not guarantee factual correctness and is presented as a research/demo project.

---

## 🧩 Core Capabilities

<table>
<tr>
<td width="50%">

### 🧠 Dual-Model Inference

- **Llama 3.2 3B Instruct** fine-tuned for agricultural QA
- **Llama 3.1 8B Instruct** as an optional higher-capacity model
- Configurable trade-off between speed and response depth

</td>
<td width="50%">

### 🎯 LLM Routing Agent

- Classifies each query before final generation
- Chooses `direct`, `rag_search`, or `web_search_tool`
- Uses structured JSON tool-call output
- Includes defensive parsing and normalization

</td>
</tr>
<tr>
<td>

### 📚 Two-Stage RAG

- Qwen3-Embedding-0.6B embeddings
- FAISS candidate retrieval
- BGE CrossEncoder reranking
- Relevance thresholding before generation

</td>
<td>

### 🌐 Live Web Retrieval

- DDGS search for current information
- HTML extraction with BeautifulSoup
- Duplicate-domain filtering
- CrossEncoder reranking with diversity limits

</td>
</tr>
<tr>
<td>

### ⚡ Streaming Inference

- Hugging Face `TextIteratorStreamer`
- Background generation thread
- Token-by-token SSE delivery
- Real-time source events

</td>
<td>

### 🛡️ Hallucination Guardrails

- Retrieval when external evidence is useful
- CrossEncoder relevance filtering
- Explicit uncertainty handling
- Prompt constraints against fabricated precise facts and dosages

</td>
</tr>
</table>

---

## 🏗️ Architecture at a Glance

```text
┌──────────────────────────────────────────────────────────────────────┐
│                              PROJECT KISAN                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USER                                                               │
│    │                                                                 │
│    ▼                                                                 │
│  ┌───────────────────────┐                                           │
│  │     Router Agent      │                                           │
│  │  Llama JSON Tool Call │                                           │
│  └───────────┬───────────┘                                           │
│              │                                                       │
│      ┌───────┼───────────────┐                                      │
│      │       │               │                                      │
│      ▼       ▼               ▼                                      │
│   DIRECT    RAG           WEB SEARCH                                │
│              │               │                                      │
│              │               ├── DDGS (15 results)                   │
│              │               ├── Fetch + Parse                       │
│              │               ├── Chunk (1500 / 200)                  │
│              │               └── BGE Rerank → Top 10                 │
│              │                                                       │
│              ├── Qwen3-Embedding-0.6B                               │
│              ├── FAISS → Top 20                                     │
│              └── BGE Rerank → Top 5                                  │
│                              │                                       │
│                              ▼                                       │
│                    ┌────────────────────┐                            │
│                    │  Context-Aware LLM │                            │
│                    │                    │                            │
│                    │ Llama 3.2 3B LoRA  │                            │
│                    │        OR          │                            │
│                    │ Llama 3.1 8B       │                            │
│                    └─────────┬──────────┘                            │
│                              │                                       │
│                              ▼                                       │
│                         FastAPI SSE                                  │
│                              │                                       │
│                              ▼                                       │
│                         Gradio UI                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔥 Fine-Tuning with QLoRA

The domain-adapted model is based on **Meta Llama 3.2 3B Instruct** and fine-tuned for agricultural question answering using **parameter-efficient QLoRA**.

### Dataset

**`kisanVaani/agriculture-qa-english-only`**

| Setting | Value |
|---|---|
| Train / validation split | 90 / 10 |
| Split seed | `42` |
| Input format | Conversational messages |
| Training objective | Supervised fine-tuning |

### Quantization

```text
load_in_4bit             = True
quantization_type        = NF4
double_quantization      = True
compute_dtype            = BF16
```

### LoRA Configuration

```text
r                  = 16
lora_alpha         = 32
lora_dropout       = 0.05
bias               = none
task_type          = CAUSAL_LM

Target modules:
  q_proj
  k_proj
  v_proj
  o_proj
```

### Training Configuration

```text
epochs                    = 2
per-device batch size     = 2
gradient accumulation      = 4
learning rate              = 2e-4
logging steps              = 10
evaluation steps           = 250
save steps                 = 250
gradient checkpointing      = enabled
BF16                       = enabled
FP16                       = disabled
reporting                  = none
early stopping patience    = 2
```

The training run used by the project reached **checkpoint-3500**.

---

## 📚 Retrieval-Augmented Generation

Project Kisan uses a **retrieve → rerank → filter** pipeline rather than relying only on embedding similarity.

### Internal Knowledge Pipeline

```text
Agricultural Documents
        │
        ▼
Qwen3-Embedding-0.6B
        │
        ▼
   FAISS Retrieval
     Top 20
        │
        ▼
BGE CrossEncoder
        │
        ▼
Score ≥ 0.5
        │
        ▼
     Top 5
        │
        ▼
 Final LLM Context
```

### RAG Configuration

| Component | Configuration |
|---|---|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| Vector store | FAISS |
| Candidate pool | 20 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Minimum rerank score | `0.5` |
| Final context | Top 5 |
| Embedding normalization | Enabled |
| Embedding batch size | 4 |
| Model execution | CUDA / FP16 |
| Attention | SDPA |

The document collection is treated as an external deployment asset; source PDFs and the generated FAISS index are not assumed to belong in the public repository.

---

## 🌐 Web Search Pipeline

For current, changing, or externally verifiable questions, the router can select the web-search tool.

```text
User Query
    │
    ▼
   DDGS
    │
    ├── Search pool: 15
    ▼
Usable webpages
    │
    ├── URL de-duplication
    ├── Domain de-duplication
    ▼
BeautifulSoup extraction
    │
    ├── Remove script/style/navigation noise
    └── Keep headings, paragraphs, list items
    │
    ▼
Chunking
1500 chars / 200 overlap
    │
    ▼
BGE CrossEncoder
    │
    ▼
Top 10 passages
max 3 passages / domain
```

This provides a separate path for information that should not rely entirely on the model's stored knowledge.

---

## 🧭 LLM Router Agent

The router makes an independent model call before final response generation.

### Three possible decisions

| Route | Use case |
|---|---|
| `direct` | Casual conversation, general explanations, or questions that do not require retrieval |
| `rag_search` | Questions that may be answered from the internal agricultural/project knowledge base |
| `web_search_tool` | Current, changing, externally verifiable, or time-sensitive information |

Example router output:

```json
{
  "name": "rag_search",
  "parameters": {
    "query": "How is wheat rust managed?"
  }
}
```

The router also handles several possible tool-call representations and attempts to recover from malformed JSON output before execution.

---

## 🛡️ Answer Generation & Guardrails

The final generation layer is explicitly prompted to prioritize useful answers without blindly copying retrieved text.

Important behaviors include:

- Understand the user's intent before answering
- Use retrieved evidence when it is relevant
- Combine evidence with general knowledge when appropriate
- Avoid blindly trusting unreliable or conflicting evidence
- Acknowledge uncertainty when necessary
- Avoid inventing names, dates, measurements, statistics, dosages, or other precise facts
- Provide practical agricultural guidance
- Answer naturally instead of exposing retrieval internals
- Match the user's language
- Adjust response depth to the question

These are **prompt- and architecture-level safeguards**, not guarantees of correctness.

---

## ⚡ Streaming Generation

Generation uses Hugging Face's `TextIteratorStreamer` and runs in a background thread so the API can forward output incrementally.

```text
Model generation thread
        │
        ▼
TextIteratorStreamer
        │
        ▼
Synchronous generator
        │
        ▼
Async queue
        │
        ▼
FastAPI SSE
        │
        ▼
Frontend token stream
```

### Generation Settings

```text
temperature          = 0.3
top_p                = 0.9
repetition_penalty   = 1.05
max_new_tokens       = 1024
do_sample             = True
```

---

## 🚀 FastAPI Backend

The backend exposes both normal JSON responses and streaming responses.

### `POST /api/chat`

Returns the complete answer after inference finishes.

### `POST /api/chat/stream`

Streams Server-Sent Events for responsive UI updates.

Supported event types:

```text
status
token
sources
error
done
```

Example token event:

```text
data: {"type":"token","content":"Wheat"}
```

The asynchronous streaming endpoint bridges the blocking model/tool pipeline with FastAPI using a background thread and `asyncio.Queue`.

---

## 🔎 Source Attribution

When retrieved information contains source metadata, the API exposes it to the client.

### Internal source

```json
{
  "type": "internal",
  "title": "agriculture_book.pdf",
  "page": 42
}
```

### Web source

```json
{
  "type": "web",
  "title": "Example article",
  "url": "https://example.com/article"
}
```

This allows the frontend to show where retrieved evidence came from rather than treating the final answer as completely opaque generation.

---

## 🖥️ Frontend

The research demo uses a **Gradio-based conversational interface** connected to the FastAPI backend.

The interface is designed around:

- Multi-turn chat
- Streaming responses
- Routing/status visibility
- Source attribution
- Retrieval-aware interactions

The backend remains independent of the UI and can be consumed by another client through its API.

---

## 📁 Repository Structure

```text
Project-Kisan/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── training/
│   └── Project_Kisan_Fine_Tuning.py
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── generation.py
│   ├── prompts.py
│   ├── rag.py
│   ├── router.py
│   └── web_search.py
│
├── data/
│   └── README.md
│
├── docs/
│   └── screenshots/
│
└── frontend/
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `training/Project_Kisan_Fine_Tuning.py` | Dataset preparation and QLoRA/SFT training |
| `backend/config.py` | Model, checkpoint, FAISS, and runtime configuration |
| `backend/generation.py` | Model loading and streamed generation |
| `backend/prompts.py` | Router and final-answer prompts |
| `backend/rag.py` | FAISS retrieval and BGE reranking |
| `backend/web_search.py` | DDGS search, webpage extraction, chunking, and reranking |
| `backend/router.py` | Tool routing, parsing, normalization, and execution |
| `backend/app.py` | FastAPI application, endpoints, SSE, and server startup |

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/<your-username>/Project-Kisan.git
cd Project-Kisan
```

### 2. Create an environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials and paths

Copy the example environment file:

```bash
cp .env.example .env
```

Set the Hugging Face token and deployment values required by your environment. Do **not** commit secrets to GitHub.

The model/checkpoint and FAISS paths can be overridden from the environment rather than being hard-coded to Kaggle paths.

### 5. Run the API

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

For the original Kaggle demo workflow, the backend can also be exposed through an ngrok tunnel using the configured `NGROK_AUTH_TOKEN`.

---

## 🔐 Security & Deployment Notes

This repository is a research/demo codebase and should be reviewed before public production deployment.

### Never commit

```text
HF tokens
ngrok tokens
.env files
model weights
LoRA checkpoints
FAISS indexes
private PDFs
Kaggle credentials
```

### Before production

- Restrict CORS origins instead of allowing all origins
- Validate and constrain externally fetched URLs
- Add SSRF protection and stricter request limits to web retrieval
- Only load trusted FAISS indexes when dangerous deserialization is enabled
- Move model/index paths into deployment configuration
- Replace the development ngrok workflow with a production hosting setup

---

## 📊 Technical Snapshot

| Area | Implementation |
|---|---|
| Domain | Agriculture / AI Assistant |
| Base LLM | Meta Llama 3.2 3B Instruct |
| Optional LLM | Meta Llama 3.1 8B Instruct |
| Fine-tuning | QLoRA / LoRA / SFTTrainer |
| Quantization | 4-bit NF4 + double quantization |
| LoRA | `r=16`, `α=32`, `dropout=0.05` |
| Dataset | `kisanVaani/agriculture-qa-english-only` |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| Vector DB | FAISS |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Web search | DDGS |
| Web parsing | BeautifulSoup |
| Backend | FastAPI |
| Streaming | Server-Sent Events |
| Frontend | Gradio |
| Deployment demo | Kaggle + ngrok |

---

## 🧪 Project Status

**Research Demo · Open Source · 2026**

The repository focuses on making the architecture and implementation reproducible while keeping large models, indexes, private credentials, and source datasets outside the Git history.

---

## 👨‍💻 Author

**Hamdan Tariq**  
BSAI, NUST

[![Portfolio](https://img.shields.io/badge/Portfolio-Hamdan%20Tariq-111827?style=flat-square&logo=googlechrome&logoColor=white)](https://hamdantariq26.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-HamdanTariq26-111827?style=flat-square&logo=github&logoColor=white)](https://github.com/HamdanTariq26)

Project Kisan was developed as an **AI research and NLP project** with the Educational Robotics Lab, NUST.

---

## 🔗 Project Links

- **Project page:** https://hamdantariq26.github.io/projects/project-kisan
- **GitHub:** https://github.com/HamdanTariq26/Project-Kisan

---

<div align="center">

### 🌱 Building more grounded AI for real-world agricultural questions.

</div>
