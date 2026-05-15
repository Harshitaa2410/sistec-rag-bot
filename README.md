# 🎓 SISTec Info Bot — RAG Chatbot

> **Sagar Institute of Science & Technology, Bhopal**  
> AI-powered college guide using Retrieval-Augmented Generation (RAG)

---

## 📌 Project Overview

SISTec Info Bot is a **RAG (Retrieval-Augmented Generation) chatbot** that answers questions about Sagar Institute of Science & Technology (SISTec), Bhopal. It retrieves relevant information from a structured knowledge base and generates accurate, source-cited answers using Google Gemini.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Sentence Transformer] ── encode query ──▶ Query Embedding
                                                │
                                                ▼
                                      [FAISS Vector Index]
                                         (cosine similarity)
                                                │
                                                ▼
                                      Top-K Relevant Chunks
                                                │
                                          + Query
                                                │
                                                ▼
                                    [Google Gemini 1.5 Flash]
                                                │
                                                ▼
                                   Answer + Source Citations
```

---

## ✅ Features

| Feature | Details |
|---|---|
| **Knowledge Base** | 11 structured documents, 45+ chunks covering all SISTec topics |
| **Chunking** | Document-aware paragraph chunking (≤400 words/chunk) with metadata |
| **Embeddings** | `all-MiniLM-L6-v2` via Sentence Transformers (384-dim) |
| **Vector Store** | FAISS IndexFlatIP (inner product = cosine on normalized vectors) |
| **LLM** | Google Gemini 1.5 Flash (free tier available) |
| **OOS Handling** | Score threshold + keyword-based out-of-scope detection |
| **Source Display** | Retrieved chunks shown with source URLs and similarity scores |
| **Caching** | FAISS index cached to disk (hash-keyed, auto-invalidated) |
| **UI** | Streamlit with SISTec branding, quick-topic buttons, test suite |

---

## 📂 Project Structure

```
sistec-rag-bot/
├── app.py                      # Main Streamlit application
├── test_rag.py                 # Accuracy & OOS test suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── sistec_knowledge_base.txt   # Structured knowledge base
└── embeddings/                 # Auto-generated FAISS index cache
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
- Visit: [aistudio.google.com](https://aistudio.google.com)
- Create a free API key (no billing required for Gemini Flash)

### 3. Launch the chatbot
```bash
streamlit run app.py
```

### 4. Run the test suite
```bash
python test_rag.py
```

---

## 📚 Knowledge Base Topics

The knowledge base (`data/sistec_knowledge_base.txt`) covers:

1. **About SGI** — Overview, colleges, vision, mission
2. **Accreditations** — NAAC, NBA, AICTE, RGPV, Industry CoEs
3. **B.Tech Programs** — All branches, eligibility, duration
4. **M.Tech Programs** — Specializations and admission
5. **MBA** — SISTec Business School details
6. **Pharmacy** — SIPTec / SIPTec-R programs
7. **Admission Process** — Documents, fees, scholarships
8. **Campus & Infrastructure** — Facilities, labs, hostels
9. **Training & Placement** — Companies, packages, training
10. **Events** — NIRMAAN, SAMADHAN, GLORY, TEDx, SIH
11. **Student Rules** — Attendance, conduct, hostel rules
12. **Scholarships** — Government and institute scholarships
13. **Digital Portals** — ERP, Alumni, T&P portal links
14. **Contact & Location** — Address, how to reach

---

## 🧪 Test Cases

### In-Scope (should return accurate answers)
- "What is the minimum attendance required?"
- "What B.Tech branches are available?"
- "How to apply for admission?"
- "Tell me about hostel facilities"
- "What are placement statistics?"
- "What events happen at SISTec?"

### Out-of-Scope (should be redirected)
- "What is the capital of France?"
- "Tell me today's cricket score"
- "How to make biryani?"
- "What is TCS stock price?"

---

## 🔧 Configuration

| Setting | Default | Description |
|---|---|---|
| `top_k` | 4 | Number of chunks retrieved per query |
| `threshold` | 0.25 | Minimum similarity score for relevance |
| Embedding model | `all-MiniLM-L6-v2` | Sentence Transformer model |
| LLM | `gemini-1.5-flash` | Google Gemini model |
| Max chunk size | 400 words | Chunking parameter |

---

## 📞 SISTec Contact

- **Website:** [sistec.ac.in](https://sistec.ac.in)
- **Phone:** 9109975760
- **Admissions:** [sistec.ac.in/admissions](https://sistec.ac.in/admissions)
- **WhatsApp:** [wa.me/919109975760](https://wa.me/919109975760)

---

## 🏆 Evaluation Criteria Met

| Requirement | Status |
|---|---|
| Document upload / predefined knowledge base | ✅ 11 structured documents |
| Proper chunking with meaningful splits | ✅ Paragraph-level, metadata-aware |
| Embeddings in FAISS | ✅ FAISS IndexFlatIP, cosine similarity |
| Relevant chunk retrieval | ✅ Top-K retrieval with scoring |
| Google Gemini API integration | ✅ Gemini 1.5 Flash |
| Streamlit web interface | ✅ Branded, responsive UI |
| Source-aware answers with citations | ✅ Chunk display with source URLs |
| OOS question handling | ✅ Score + keyword based detection |
| Basic accuracy testing | ✅ `test_rag.py` with 5 test categories |
