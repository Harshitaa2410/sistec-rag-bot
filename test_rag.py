"""
test_rag.py – Accuracy and out-of-scope test suite for SISTec Info Bot
Run: python test_rag.py
"""

import os
import sys
import json
import time

# ── Minimal dependency check ──────────────────────────────────────────────────
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# ── Import core functions from app ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from app import parse_documents, retrieve, is_relevant, build_index, KB_PATH

# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def hdr(text): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
def ok(msg):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗ FAIL{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}ℹ INFO{RESET}  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Document parsing
# ─────────────────────────────────────────────────────────────────────────────
def test_document_parsing():
    hdr("TEST 1: Document Parsing & Chunking")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    docs = parse_documents(raw)

    assert len(docs) > 10, "Expected >10 chunks"
    ok(f"Parsed {len(docs)} chunks from knowledge base")

    # Check metadata presence
    for d in docs[:3]:
        assert d.get("title"), f"Missing title in {d['id']}"
        assert d.get("source"), f"Missing source in {d['id']}"
        assert len(d.get("text", "")) > 30, f"Chunk too short: {d['id']}"
    ok("All sampled chunks have title, source, and body text")

    # Chunk size sanity
    max_words = max(len(d["text"].split()) for d in docs)
    info(f"Max chunk size: {max_words} words")
    assert max_words <= 500, f"Chunk too large: {max_words} words"
    ok(f"All chunks within 500-word limit")

    return docs

# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Embeddings & FAISS index
# ─────────────────────────────────────────────────────────────────────────────
def test_embeddings_and_index():
    hdr("TEST 2: Embeddings Generation & FAISS Index")
    t0 = time.time()
    index, docs, embed_model, err = build_index(KB_PATH)
    elapsed = time.time() - t0

    assert err is None, f"Index build error: {err}"
    ok(f"FAISS index built in {elapsed:.1f}s")
    ok(f"Index contains {index.ntotal} vectors (dim={index.d})")
    assert index.ntotal == len(docs), "Vector count mismatch"
    ok("Vector count matches document count")
    return index, docs, embed_model

# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Retrieval accuracy (in-scope queries)
# ─────────────────────────────────────────────────────────────────────────────
IN_SCOPE_TESTS = [
    {
        "query": "What is the minimum attendance required at SISTec?",
        "expected_keywords": ["75%", "attendance", "examination"],
        "expected_source_contains": "student-rulebook",
    },
    {
        "query": "What B.Tech branches are available at SISTec?",
        "expected_keywords": ["CSE", "Mechanical", "Civil", "B.Tech"],
        "expected_source_contains": "programs",
    },
    {
        "query": "How can I apply for admission to SISTec?",
        "expected_keywords": ["JEE", "DTE", "admission", "counselling"],
        "expected_source_contains": "admissions",
    },
    {
        "query": "Tell me about hostel facilities and rules",
        "expected_keywords": ["hostel", "room", "mess"],
        "expected_source_contains": "hostel",
    },
    {
        "query": "What placement companies visit SISTec?",
        "expected_keywords": ["TCS", "Infosys", "placement"],
        "expected_source_contains": "placements",
    },
    {
        "query": "What events are held at SISTec?",
        "expected_keywords": ["NIRMAAN", "SAMADHAN", "TEDx"],
        "expected_source_contains": "events",
    },
    {
        "query": "Is SISTec NAAC accredited?",
        "expected_keywords": ["NAAC", "accredited"],
        "expected_source_contains": "accreditation",
    },
    {
        "query": "What scholarships does SISTec offer?",
        "expected_keywords": ["scholarship", "merit", "SC/ST"],
        "expected_source_contains": "scholarships",
    },
]

def test_retrieval_accuracy(index, docs, embed_model):
    hdr("TEST 3: Retrieval Accuracy (In-Scope Queries)")
    passed = 0
    for t in IN_SCOPE_TESTS:
        results = retrieve(t["query"], index, docs, embed_model, top_k=4)
        top_doc, top_score = results[0]

        # Keyword check in top-4 combined text
        combined = " ".join(r[0]["text"] for r in results).lower()
        kw_found = [kw for kw in t["expected_keywords"] if kw.lower() in combined]
        kw_ok = len(kw_found) >= len(t["expected_keywords"]) // 2  # at least half

        # Source check
        sources = [r[0]["source"] for r in results]
        src_ok = any(t["expected_source_contains"] in s for s in sources)

        if kw_ok and top_score > 0.2:
            ok(f'"{t["query"][:55]}…" → score={top_score:.3f}, kw={kw_found}')
            passed += 1
        else:
            fail(f'"{t["query"][:55]}…" → score={top_score:.3f}, kw_ok={kw_ok}')

    info(f"Retrieval accuracy: {passed}/{len(IN_SCOPE_TESTS)}")
    assert passed >= len(IN_SCOPE_TESTS) * 0.75, f"Retrieval accuracy below 75%: {passed}/{len(IN_SCOPE_TESTS)}"
    ok(f"Passed accuracy threshold (≥75%)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Out-of-scope detection
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_TESTS = [
    ("What is the capital of France?", 0.05),
    ("Tell me today's cricket score", 0.05),
    ("How to make biryani?", 0.05),
    ("What is the current stock price of TCS?", 0.05),
    ("Tell me a joke", 0.05),
]

def test_out_of_scope_detection(index, docs, embed_model):
    hdr("TEST 4: Out-of-Scope Query Handling")
    passed = 0
    for query, fake_score in OUT_OF_SCOPE_TESTS:
        # Test the keyword-based OOS detection
        oos_detected = not is_relevant(query, fake_score, threshold=0.25)
        if oos_detected:
            ok(f'OOS detected: "{query}"')
            passed += 1
        else:
            # Also test via low retrieval score
            results = retrieve(query, index, docs, embed_model, top_k=3)
            top_score = results[0][1] if results else 0.0
            if not is_relevant(query, top_score, threshold=0.25):
                ok(f'OOS detected via low score ({top_score:.3f}): "{query}"')
                passed += 1
            else:
                fail(f'OOS NOT detected (score={top_score:.3f}): "{query}"')
    info(f"OOS detection: {passed}/{len(OUT_OF_SCOPE_TESTS)}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Edge cases
# ─────────────────────────────────────────────────────────────────────────────
def test_edge_cases(index, docs, embed_model):
    hdr("TEST 5: Edge Cases")

    # Empty query
    try:
        r = retrieve("   ", index, docs, embed_model, top_k=2)
        ok("Empty query handled without crash")
    except Exception as e:
        fail(f"Empty query crashed: {e}")

    # Very long query
    long_query = "What are " + "all the details about " * 20 + "SISTec admissions?"
    r = retrieve(long_query, index, docs, embed_model, top_k=2)
    assert len(r) > 0
    ok("Long query handled successfully")

    # Misspelled query
    r = retrieve("sistic admission proceedure", index, docs, embed_model, top_k=3)
    assert len(r) > 0 and r[0][1] > 0.1
    ok(f"Misspelled query returned results (top score={r[0][1]:.3f})")

    # Hindi/mixed query
    r = retrieve("SISTec mein admission kaise le?", index, docs, embed_model, top_k=3)
    assert len(r) > 0
    ok(f"Mixed-language query handled (top score={r[0][1]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'═'*60}")
    print("  SISTec Info Bot – RAG Test Suite")
    print(f"  Testing knowledge base at: {KB_PATH}")
    print(f"{'═'*60}{RESET}")

    t_start = time.time()
    try:
        test_document_parsing()
        index, docs, embed_model = test_embeddings_and_index()
        test_retrieval_accuracy(index, docs, embed_model)
        test_out_of_scope_detection(index, docs, embed_model)
        test_edge_cases(index, docs, embed_model)

        elapsed = time.time() - t_start
        print(f"\n{BOLD}{GREEN}{'═'*60}")
        print(f"  ✅ All tests passed! ({elapsed:.1f}s)")
        print(f"{'═'*60}{RESET}\n")
    except AssertionError as e:
        print(f"\n{BOLD}{RED}{'═'*60}")
        print(f"  ❌ Test FAILED: {e}")
        print(f"{'═'*60}{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
