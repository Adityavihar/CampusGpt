# CampusGPT — AI-Powered University Information Assistant

**CampusGPT** is an intelligent, retrieval-augmented generation (RAG) assistant designed for **VIT-AP University**. It enables students, prospective applicants, and faculty to obtain accurate, institution-verified information through natural language queries.

Beyond simple keyword search or narrow attendance calculators, CampusGPT serves as a comprehensive institutional knowledge platform answering questions regarding academic programs, admissions criteria, academic regulations, Fully Flexible Credit System (FFCS), 75% attendance policies, tuition fees, scholarships, hostels, and campus facilities.

---

## 1. Key Features

- **Document-Grounded QA**: Answers are generated strictly from authoritative university documents. If information is unavailable in the knowledge base, the assistant explicitly reports that the information cannot be found, preventing hallucinations.
- **Clean Direct Presentation**: Answers are structured cleanly with bullet points, bold key terms, and summary tables without cluttered footnotes or sources displayed below.
- **Conversational Context & Follow-Ups**: Remembers conversation context to resolve follow-up queries (e.g., "What B.Tech programs exist?" followed by "What about M.Tech?").
- **Selective OCR Fallback**: Employs PyMuPDF for high-speed digital text extraction and invokes Tesseract OCR only on scanned or image-only pages, preserving exact page alignment without redundant processing.
- **Idempotent Ingestion Pipeline**: File-hash (SHA-256) tracking prevents duplicate chunks and wasted API calls when re-indexing documents.
- **Batch Embedding Generation**: Uses Jina AI (`jina-embeddings-v3`, 1024 dimensions) with batched requests, exponential backoff, and retry handling.
- **Persistent Vector Store**: ChromaDB vector store configured with cosine similarity and metadata filters for document categories.
- **Attendance Calculator (Supporting Feature)**: Parses student attendance reports and screenshots, validates totals, detects shortage risk (<75%), and calculates exact recovery classes required to reach 75% eligibility.
- **Modern Responsive Web Interface**: Clean UI built with vanilla HTML5, CSS3, and JavaScript featuring suggested prompts, markdown formatting, and attendance modal.

---

## 2. System Architecture

### A. RAG Query Pipeline
```
Student Question
       │
       ▼
 Flask Web API (/api/chat)
       │
       ▼
Query Reformulation (Incorporating Session History)
       │
       ▼
Jina Embeddings API (`jina-embeddings-v3`, task="retrieval.query", 1024-dim)
       │
       ▼
ChromaDB Vector Store (`vectorstore/chroma`)
       │ (Cosine Distance Search + Category Filtering)
       ▼
Top Relevant University Document Chunks + Source Metadatas
       │
       ▼
Groq LLM (`openai/gpt-oss-20b`)
  ├─ System Prompt: Strict grounding, no hallucinations
  ├─ Context: Excerpts + Verified Document Sources
  └─ Output: Clean, direct Markdown answer
       │
       ▼
Formatted Answer rendered cleanly in Chat Interface
```

### B. Document Ingestion Pipeline
```
University PDF Documents (`data/documents/*.pdf`)
       │
       ▼
PyMuPDF Text Parser (`services/document_parser.py`)
       │
       ├── Page has digital text? ───► Extract text directly
       │
       └── Page is scanned/image? ───► Targeted Tesseract OCR (`services/ocr_service.py`)
       │
       ▼
Clean Text & Sliding-Window Chunking (500 chars, 50 overlap)
       │
       ▼
Batch Embedding Generation (Jina API, batch size 32)
       │
       ▼
Idempotent ChromaDB Upsert + Catalog Tracking (`data/catalog.json`)
```

---

## 3. Technology Stack

| Layer | Component | Version / Description |
|---|---|---|
| **Backend** | Python / Flask | Flask 3.1.3, Werkzeug 3.1.8, Python 3.11 |
| **Vector DB** | ChromaDB | ChromaDB 1.5.9 (Persistent SQLite3 + HNSW) |
| **Embeddings** | Jina AI API | `jina-embeddings-v3` (1024-dimensional vectors) |
| **LLM Provider** | Groq Cloud | `openai/gpt-oss-20b` (low reasoning, temperature 0.1) |
| **PDF Processing** | PyMuPDF (fitz) | 1.28.2 (Fast digital text extraction & pixmaps) |
| **OCR Fallback** | Tesseract + PIL | Tesseract OCR 5.x, `pytesseract` 0.3.13, Pillow 13.0 |
| **PDF Images** | pdf2image + Poppler | Poppler 26.09 (Windows) / `poppler-utils` (Linux) |
| **Frontend** | HTML5 / CSS / Vanilla JS | Responsive, Markdown parser, modal dialogs |
| **Testing** | Pytest | Pytest 9.2.0 (26 automated test cases) |

---

## 4. Project Directory Structure

```
CampusAI/
├── app.py                         # Flask web application & API routes
├── requirements.txt               # Production Python dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusion rules
├── README.md                      # Comprehensive project documentation
├── data/
│   ├── catalog.json               # Document metadata, SHA-256 hashes & categories
│   ├── documents/                 # Authoritative university PDF collection
│   │   ├── VITAP_Public_Self_Disclosure.pdf
│   │   ├── VITAP_Academic_Regulations_and_FFCS.pdf
│   │   ├── VITAP_Programs_and_Admissions_Guide.pdf
│   │   ├── VITAP_Campus_Facilities_and_Hostels.pdf
│   │   └── VITAP_Fee_Structure_and_Scholarships.pdf
│   └── uploads/                   # Temporary folder for attendance uploads (auto-purged)
├── services/
│   ├── document_parser.py         # PyMuPDF extractor with per-page OCR fallback
│   ├── ocr_service.py             # Tesseract & pdf2image OCR integration
│   ├── chunking.py                # Sliding-window document chunking
│   ├── embeddings.py              # Jina AI batched embeddings
│   ├── vector_store.py            # ChromaDB persistent store & idempotent indexing
│   ├── retriever.py               # Semantic search & formatted chunk retrieval
│   ├── llm.py                     # Groq LLM grounded prompt & answer generation
│   ├── ingestion_service.py       # Multi-document cataloging & batch ingestion
│   └── attendance_service.py      # Attendance parsing, validation & recovery math
├── templates/
│   └── index.html                 # Responsive chat UI & attendance tool
├── scripts/
│   ├── build_knowledge_base.py    # Generates authoritative university PDFs
│   └── e2e_validation.py          # End-to-end live testing script
├── tests/
│   ├── test_api.py                # Flask API test cases
│   ├── test_attendance.py         # Attendance parsing & recovery calculation tests
│   ├── test_chunking.py           # Chunking boundary tests
│   ├── test_document_parser.py    # PDF parsing & text cleaning tests
│   ├── test_llm.py                # LLM prompt grounding tests
│   ├── test_ocr_pipeline.py       # Scanned and mixed PDF OCR tests
│   ├── test_retriever.py          # Vector retrieval tests
│   └── test_vector_store.py       # ChromaDB idempotent indexing tests
└── vectorstore/
    └── chroma/                    # ChromaDB persistent database files
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Tesseract OCR:
  - **Windows**: Install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
  - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y tesseract-ocr poppler-utils`.
- Poppler (Optional on Windows, handled natively via PyMuPDF pixmap fallback):
  - **Windows**: Extracted to `C:\poppler\poppler-26.09.0\Library\bin`.

### Step 1: Clone and Create Virtual Environment
```bash
git clone <repo-url> F:\CampusAI
cd F:\CampusAI

python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate    # Linux / macOS
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux
```
Edit `.env` to include your API keys:
```env
JINA_API_KEY=your_jina_api_key
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=secure_random_flask_secret_key
PORT=5000
FLASK_DEBUG=false
```

---

## 6. How to Run the Application

### Start Development Server
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

### Run Automated Tests
```bash
pytest tests/
```
All 26 test cases test unit logic, API endpoints, OCR fallback, vector store, and attendance calculation with external APIs safely mocked.

### Run Live End-to-End Validation
```bash
python scripts/e2e_validation.py
```

---

## 7. How to Add and Index University Documents

1. Place any official university PDF file in `data/documents/` (e.g., `data/documents/VITAP_Placement_Report_2025.pdf`).
2. Optionally update `data/catalog.json` with the document title and category.
3. Trigger ingestion via Python or API:
   - **CLI**:
     ```bash
     python -c "from services.ingestion_service import index_all_documents; print(index_all_documents())"
     ```
   - **API POST**:
     ```bash
     curl -X POST http://127.0.0.1:5000/api/index-document -H "Content-Type: application/json" -d "{\"all\": true}"
     ```

---

## 8. API Endpoints

| Endpoint | Method | Parameters / Body | Description |
|---|---|---|---|
| `/` | `GET` | — | Serves the CampusGPT chat UI. |
| `/api/health` | `GET` | — | Returns API health, ChromaDB chunk count, and catalog status. |
| `/api/documents` | `GET` | — | Lists all indexed university documents and metadata. |
| `/api/search` | `GET` | `?q=<query>&limit=5` | Semantic vector search returning top matching excerpts with distances. |
| `/api/chat` | `POST` | `{"question": "..."}` | Main QA endpoint. Retrieves context, calls Groq LLM, updates history, returns answer + citations. |
| `/api/clear-chat`| `POST` | — | Clears user session chat history. |
| `/api/index-document` | `POST` | `{"all": true, "force": false}` | Ingests and indexes PDFs into ChromaDB idempotently. |
| `/api/attendance/extract` | `POST` | Multipart `file` or JSON `{"text": "..."}` | Parses attendance records, checks 75% rule, and computes recovery classes. |

---

## 9. Attendance Calculation Logic

Under VIT-AP regulations, students must maintain a minimum **75% attendance** in each registered course:
$$\text{Percentage} = \left(\frac{\text{Attended Classes}}{\text{Total Conducted Classes}}\right) \times 100$$

When $\text{Percentage} < 75\%$:
The student must attend $x$ consecutive future classes without missing any to reach 75%:
$$\frac{\text{Attended} + x}{\text{Total} + x} \ge 0.75 \implies x = \lceil 3 \cdot \text{Total} - 4 \cdot \text{Attended} \rceil$$

When $\text{Percentage} \ge 75\%$:
The student can miss $y$ classes without dropping below 75%:
$$\frac{\text{Attended}}{\text{Total} + y} \ge 0.75 \implies y = \lfloor \frac{4 \cdot \text{Attended}}{3} - \text{Total} \rfloor$$

---

## 10. Security & Production Considerations

1. **Credentials**: Secrets (`JINA_API_KEY`, `GROQ_API_KEY`, `SECRET_KEY`) are kept strictly in `.env` and excluded via `.gitignore`.
2. **File Processing Privacy**: Uploaded student attendance documents are processed in-memory / temporary files and deleted immediately in `finally` blocks.
3. **Prompt Injection & Grounding**: LLM instructions strictly isolate system prompts from user queries and treat retrieved documents as untrusted data.
4. **Production Deployment**:
   - For Linux production using Gunicorn:
     ```bash
     gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
     ```
   - For Windows production using Waitress:
     ```bash
     pip install waitress
     waitress-serve --port=5000 wsgi:app
     ```
   - For Docker Container Deployment:
     ```bash
     docker build -t campusgpt .
     docker run -d -p 5000:5000 --env-file .env -v campus_chroma:/app/vectorstore/chroma campusgpt
     ```
   - Place behind an Nginx reverse proxy with HTTPS enabled in cloud environments.
   - Vector database storage directory `vectorstore/chroma/` should be mounted to a persistent volume when deploying in Docker or Kubernetes.
