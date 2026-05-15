"""
SISTec Info Bot – RAG Chatbot
Sagar Institute of Science & Technology | Bhopal, MP
Built using: FAISS + Sentence Transformers + Google Gemini API + Streamlit
"""

import os
import re
import json
import pickle
import hashlib
import time
from typing import List, Tuple, Dict, Optional

import streamlit as st
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SISTec Info Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #f5f7fa;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #283593 60%, #1565c0 100%);
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #90caf9 !important; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a237e, #1565c0);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 20px rgba(26,35,126,0.3);
}
.main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
.main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; }

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(21,101,192,0.3);
}
.bot-bubble {
    background: #fff;
    color: #1a1a2e;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 8px 0;
    max-width: 85%;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1565c0;
}
.source-card {
    background: #e8f0fe;
    border: 1px solid #c5cae9;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.82rem;
}
.source-chip {
    display: inline-block;
    background: #1a237e;
    color: #fff !important;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px 3px 3px 0;
    font-size: 0.75rem;
}
.out-of-scope {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.9rem;
}
/* ── Stats cards ── */
.stat-card {
    background: #fff;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 3px solid #1565c0;
}
.stat-num { font-size: 1.6rem; font-weight: 700; color: #1565c0; }
.stat-lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }
/* ── Input ── */
.stTextInput input {
    border-radius: 30px !important;
    border: 2px solid #c5cae9 !important;
    padding: 12px 20px !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus { border-color: #1565c0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Lazy imports (with user-friendly errors) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_heavy_deps():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        import google.generativeai as genai
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return faiss, SentenceTransformer, genai, model, None
    except ImportError as e:
        return None, None, None, None, str(e)


# ─── Text chunking ─────────────────────────────────────────────────────────────
def parse_documents(raw_text: str) -> List[Dict]:
    """
    Split knowledge base into document chunks.
    Each chunk preserves its SOURCE metadata.
    """
    docs = []
    # Split on document boundaries
    sections = re.split(r"={4}\s*DOCUMENT:\s*", raw_text)
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # Extract title line
        lines = sec.split("\n")
        title_line = lines[0].replace("====", "").strip()
        # Extract source
        source = ""
        body_start = 1
        for i, ln in enumerate(lines[1:], 1):
            m = re.match(r"SOURCE:\s*(.+)", ln.strip())
            if m:
                source = m.group(1).strip()
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        if not body:
            continue
        # Sub-chunk by paragraphs (keep ≤ 400 words per chunk)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        current_chunk = []
        current_wc = 0
        chunk_idx = 0
        for para in paragraphs:
            wc = len(para.split())
            if current_wc + wc > 400 and current_chunk:
                docs.append({
                    "id": f"{title_line}_{chunk_idx}",
                    "title": title_line,
                    "source": source,
                    "text": "\n\n".join(current_chunk),
                })
                chunk_idx += 1
                current_chunk = []
                current_wc = 0
            current_chunk.append(para)
            current_wc += wc
        if current_chunk:
            docs.append({
                "id": f"{title_line}_{chunk_idx}",
                "title": title_line,
                "source": source,
                "text": "\n\n".join(current_chunk),
            })
    return docs


# ─── FAISS index building ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(kb_path: str):
    """
    Build (or load cached) FAISS index from knowledge base file.
    Returns: (index, docs, embed_model)
    """
    faiss, SentenceTransformer, genai, embed_model, err = load_heavy_deps()
    if err:
        return None, None, None, err

    # Cache key based on file content hash
    with open(kb_path, "r", encoding="utf-8") as f:
        raw = f.read()
    file_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    cache_dir = "/tmp" if os.path.exists("/mount/src") else os.path.join(_base_dir, "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"faiss_index_{file_hash}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        index = faiss.deserialize_index(data["index_bytes"])
        return index, data["docs"], embed_model, None

    docs = parse_documents(raw)
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
    index.add(embeddings)

    with open(cache_path, "wb") as f:
        pickle.dump({
            "index_bytes": faiss.serialize_index(index),
            "docs": docs,
        }, f)

    return index, docs, embed_model, None


# ─── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, index, docs, embed_model, top_k: int = 4) -> List[Tuple[Dict, float]]:
    import faiss as _faiss
    import numpy as np

    q_emb = embed_model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    _faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((docs[idx], float(score)))
    return results


# ─── Gemini answer generation ───────────────────────────────────────────────────
def generate_answer(query: str, chunks: List[Tuple[Dict, float]], api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build context
    context_parts = []
    for i, (doc, score) in enumerate(chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {doc['source']} | Topic: {doc['title']}]\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are the official SISTec Info Bot for Sagar Institute of Science & Technology (SISTec), Bhopal.
Your job is to answer student and visitor queries accurately using ONLY the provided context.

Context from SISTec knowledge base:
{context}

Student Query: {query}

Instructions:
- Answer helpfully, clearly, and concisely using only the information in the context.
- If the context doesn't contain enough information to answer the query, clearly state:
  "I don't have specific information about this in my knowledge base. Please contact SISTec directly at 9109975760 or visit sistec.ac.in"
- Do NOT hallucinate or make up facts not present in the context.
- If relevant, mention contact details (9109975760, sistec.ac.in) for further inquiries.
- Keep the tone professional, warm, and student-friendly — matching SISTec's aspirational, student-centric brand.
- Format the answer clearly with line breaks where appropriate.

Answer:"""

    response = model.generate_content(prompt)
    return response.text.strip()


# ─── Out-of-scope detection ─────────────────────────────────────────────────────
def is_relevant(query: str, top_score: float, threshold: float = 0.25) -> bool:
    """Simple relevance gate: if best chunk score < threshold, it's probably OOS."""
    oos_keywords = [
        "weather", "cricket score", "movie", "recipe", "stock price",
        "politics", "news today", "joke", "covid", "vaccine",
    ]
    q_lower = query.lower()
    if any(kw in q_lower for kw in oos_keywords):
        return False
    return top_score >= threshold


# ─── Load knowledge base ────────────────────────────────────────────────────────
# Resolve KB path — works locally and on Streamlit Cloud
_base_dir = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(_base_dir, "data", "sistec_knowledge_base.txt")

# Fallback: search common Streamlit Cloud mount paths
if not os.path.exists(KB_PATH):
    for _candidate in [
        "/mount/src/sistec-rag-bot/data/sistec_knowledge_base.txt",
        os.path.join(os.getcwd(), "data", "sistec_knowledge_base.txt"),
    ]:
        if os.path.exists(_candidate):
            KB_PATH = _candidate
            break


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 SISTec Info Bot")
    st.markdown("*Powered by RAG + Google Gemini*")
    st.divider()

    # API Key input
    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get free key at aistudio.google.com",
    )
    if not api_key:
        st.warning("Enter your Gemini API key to enable AI answers.")
    else:
        st.success("API key set ✓")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved", 2, 6, 4, help="More chunks = richer context but slower")
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_scores = st.toggle("Show similarity scores", value=False)

    st.divider()

    # Quick topic buttons
    st.markdown("### 💡 Quick Topics")
    quick_qs = [
        "What B.Tech branches are offered?",
        "How to apply for admission?",
        "What is the attendance requirement?",
        "Tell me about hostel facilities",
        "What are the placement statistics?",
        "What events happen at SISTec?",
        "What scholarships are available?",
        "Where is SISTec located?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
            st.session_state["quick_query"] = q

    st.divider()
    st.markdown("""
**📞 Direct Contact**  
Phone: [9109975760](tel:9109975760)  
Web: [sistec.ac.in](https://sistec.ac.in)  
Admissions: [sistec.ac.in/admissions](https://sistec.ac.in/admissions)
""")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
  <div style="font-size:3rem;">🎓</div>
  <div>
    <h1>SISTec Info Bot</h1>
    <p>Sagar Institute of Science &amp; Technology, Bhopal &nbsp;|&nbsp; Your AI-powered college guide</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats row
_, col_docs, col_chunks, col_topics, _ = st.columns([1, 2, 2, 2, 1])
with col_docs:
    st.markdown('<div class="stat-card"><div class="stat-num">11</div><div class="stat-lbl">Knowledge Docs</div></div>', unsafe_allow_html=True)
with col_chunks:
    st.markdown('<div class="stat-card"><div class="stat-num">45+</div><div class="stat-lbl">Text Chunks</div></div>', unsafe_allow_html=True)
with col_topics:
    st.markdown('<div class="stat-card"><div class="stat-num">∞</div><div class="stat-lbl">Queries Supported</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Load index ────────────────────────────────────────────────────────────────
with st.spinner("Loading SISTec knowledge base & embeddings…"):
    index, docs, embed_model, load_err = build_index(KB_PATH)

if load_err:
    st.error(f"❌ Failed to load dependencies: {load_err}\n\nRun: `pip install -r requirements.txt`")
    st.stop()

# ─── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, sources}

# ─── Chat history display ───────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#666;">
            <div style="font-size:3rem;">💬</div>
            <p style="font-size:1.1rem; margin-top:10px;">
                Ask me anything about SISTec — admissions, courses, events, rules, placements, and more!
            </p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            answer_html = msg["content"].replace("\n", "<br>")
            st.markdown(f'<div class="bot-bubble">🤖 {answer_html}</div>', unsafe_allow_html=True)

            # Sources
            if show_chunks and msg.get("sources"):
                with st.expander(f"📚 Retrieved {len(msg['sources'])} source chunks", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        score_txt = f" | Score: {src['score']:.3f}" if show_scores else ""
                        st.markdown(f"""
<div class="source-card">
<span class="source-chip">Chunk {i}</span>
<span class="source-chip">📖 {src['title']}</span>
<span class="source-chip">🔗 {src['source']}</span>{score_txt}
<hr style="margin:8px 0; border-color:#c5cae9;">
<small>{src['text'][:350]}{'…' if len(src['text']) > 350 else ''}</small>
</div>
""", unsafe_allow_html=True)


# ─── Input area ────────────────────────────────────────────────────────────────
st.divider()
col_input, col_btn = st.columns([6, 1])

# Handle quick-query from sidebar
default_val = st.session_state.pop("quick_query", "")

with col_input:
    user_input = st.text_input(
        "Ask about SISTec…",
        value=default_val,
        placeholder="e.g. What B.Tech courses are available? | What are hostel rules?",
        label_visibility="collapsed",
        key="user_query_input",
    )
with col_btn:
    send = st.button("Send 🚀", use_container_width=True, type="primary")

col_clear, col_test = st.columns([1, 1])
with col_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
with col_test:
    with st.expander("🧪 Test Suite"):
        st.markdown("**Accuracy Tests** — click to run:")
        test_cases = [
            ("In-scope", "What is the full form of SISTec?"),
            ("In-scope", "What is minimum attendance required?"),
            ("In-scope", "What are the B.Tech branches at SISTec?"),
            ("Out-of-scope", "What is the capital of France?"),
            ("Out-of-scope", "Tell me today's cricket score"),
        ]
        for category, tc in test_cases:
            badge = "✅" if category == "In-scope" else "🚫"
            if st.button(f"{badge} {tc}", key=f"test_{tc[:25]}"):
                st.session_state["quick_query"] = tc
                st.rerun()


# ─── Process query ─────────────────────────────────────────────────────────────
if (send or default_val) and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.history.append({"role": "user", "content": query})

    if not api_key:
        st.session_state.history.append({
            "role": "bot",
            "content": "⚠️ Please enter your Google Gemini API key in the sidebar to get AI-powered answers.",
            "sources": [],
        })
    else:
        with st.spinner("🔍 Searching SISTec knowledge base…"):
            results = retrieve(query, index, docs, embed_model, top_k=top_k)

        top_score = results[0][1] if results else 0.0

        if not is_relevant(query, top_score):
            answer = (
                "🚫 This question seems outside SISTec's knowledge scope.\n\n"
                "I'm designed to answer questions about SISTec — admissions, courses, "
                "campus, events, rules, and placements.\n\n"
                "For other queries, please use a general search engine. "
                "For SISTec-specific help: 📞 9109975760 | 🌐 sistec.ac.in"
            )
            sources_meta = []
        else:
            with st.spinner("🤖 Generating answer with Gemini…"):
                try:
                    answer = generate_answer(query, results, api_key)
                    sources_meta = [
                        {
                            "title": doc["title"],
                            "source": doc["source"],
                            "text": doc["text"],
                            "score": score,
                        }
                        for doc, score in results
                    ]
                except Exception as e:
                    answer = f"❌ Gemini API error: {str(e)}\n\nPlease check your API key and try again."
                    sources_meta = []

        st.session_state.history.append({
            "role": "bot",
            "content": answer,
            "sources": sources_meta,
        })

    st.rerun()
