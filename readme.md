# HR Policy Chatbot — Implementation Blueprint
## Tailored to Your Existing Project

---

## What You Already Have (and What's Good About It)

Your scraped JSON is actually excellent. The structure in `all_policies_scraped.json` already gives you:

- `policyId` (e.g., "PD-58") — links national → state variations
- `category` (e.g., "Associate Support") — useful for routing
- `stateVariations[]` with `stateName`, `stateCode`, `url`, `content.text`, `effectiveDate`
- National content nested under `content.text`

Your JSONL chunks already have: `chunk_id`, `policy_name`, `policy_id`, `category`, `state`, `state_code`, `is_state_variation`, `effective_date`, `chunk_index`, `total_chunks`, `text`

**What's missing from your chunk metadata (add these):**

```python
# Fields to add during re-chunking:
"applies_to": ["all"],           # or ["salaried", "hourly", "field", "store_manager", "home_office"]
"section_heading": "Eligibility", # extracted from the text structure
"parent_chunk_id": "...",         # for parent-child retrieval
"parent_policy_id": "PD-58",     # always points to the national policy_id
                                  # (same as policy_id for national, links back for state variants)
```

The `applies_to` field is the KEY to making the system ask "are you salaried or hourly?" dynamically.
You'll populate this semi-automatically (see Phase 1 below).

---

## File-by-File Implementation Plan

Here's every file you need, what it does, and pseudocode for each. You can keep your existing project structure and rewrite the Python scripts.

```
POLICY CHATBOT 2/
├── config.py              # All configuration in one place
├── chunk_policies.py      # REWRITE: parent-child chunking + metadata enrichment
├── embed_and_store.py     # REWRITE: hybrid indexing (ChromaDB + BM25)
├── rag_retrieval.py       # REWRITE: hybrid search + reranking pipeline
├── rag_generation.py      # REWRITE: agentic orchestration with tools
├── rag_store.py           # KEEP/MODIFY: ChromaDB setup
├── rag_config.py          # MERGE INTO config.py
├── app.py                 # MODIFY: Flask routes + session management
├── audience_utils.py      # EVOLVE: role/audience detection logic
├── state_utils.py         # KEEP: state detection/normalization
├── logging_db.py          # KEEP: conversation logging
├── requirements.txt       # UPDATE
├── all_policies_scraped.json     # YOUR DATA (don't touch)
├── chunks/                       # REGENERATE with new chunking
│   ├── chunks_all.jsonl
│   ├── chunks_national.jsonl
│   └── chunks_national_plus_state.jsonl
├── chroma_db/                    # REGENERATE with new embeddings
├── parent_store/                 # NEW: parent chunk storage
│   └── parents.db                # SQLite for parent chunks
└── cache/                        # NEW: semantic cache
    └── cache.db
```

---

## config.py — Single Source of Truth

```python
"""
All configuration lives here. Import from everywhere else.
"""
import os

# === LLM ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"                          # agent reasoning
LLM_MODEL_FAST = "gpt-4o-mini"                # chunk enrichment, cheap tasks
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# === ChromaDB ===
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "hr_policies"

# === Chunking ===
PARENT_CHUNK_SIZE = 1200       # tokens — full policy sections
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250         # tokens — precise retrieval units
CHILD_CHUNK_OVERLAP = 30
PARENT_STORE_PATH = "./parent_store/parents.db"

# === Retrieval ===
HYBRID_DENSE_WEIGHT = 0.5     # tune: higher = more semantic
HYBRID_SPARSE_WEIGHT = 0.5    # tune: higher = more keyword
INITIAL_RETRIEVE_K = 30       # candidates from each retriever
RERANK_TOP_K = 5              # final docs sent to LLM
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RELEVANCE_THRESHOLD = 0.3     # below this, admit uncertainty

# === Metadata: role/audience categories ===
ROLE_CATEGORIES = [
    "all", "salaried", "hourly", "field",
    "store_manager", "home_office", "management", "part_time"
]

# === Data paths ===
SCRAPED_JSON_PATH = "./all_policies_scraped.json"
CHUNKS_OUTPUT_DIR = "./chunks"
```

---

## chunk_policies.py — Parent-Child Chunking with Metadata

This is a full rewrite. The goal: take your `all_policies_scraped.json`, produce parent chunks (stored in SQLite) and child chunks (stored in JSONL and later embedded into ChromaDB), with rich metadata on every chunk.

```python
"""
chunk_policies.py

Reads all_policies_scraped.json
Produces:
  - Parent chunks → SQLite (parent_store/parents.db)
  - Child chunks  → chunks/chunks_enriched.jsonl
  
Parent-child strategy:
  1. Split each policy's text on section headings (H2/H3 patterns) → parent chunks
  2. Split each parent into smaller child chunks → child chunks
  3. Each child stores its parent_chunk_id for retrieval-time expansion
"""

import json
import sqlite3
import hashlib
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *


def init_parent_store(db_path):
    """
    Create SQLite DB for parent chunks.
    Schema: parent_chunk_id TEXT PK, policy_id TEXT, text TEXT, metadata JSON
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_chunk_id TEXT PRIMARY KEY,
            policy_id TEXT,
            policy_name TEXT,
            state TEXT,
            state_code TEXT,
            category TEXT,
            section_heading TEXT,
            text TEXT,
            metadata_json TEXT
        )
    """)
    conn.commit()
    return conn


def make_chunk_id(policy_id, state_code, chunk_index, level="child"):
    """Deterministic chunk ID: policy__PD-58__CA__child-3"""
    state_part = state_code or "national"
    return f"policy__{policy_id}__{state_part}__{level}-{chunk_index}"


def extract_section_headings(text):
    """
    Try to split policy text on section-like patterns.
    Walmart policies tend to have patterns like:
      "Purpose", "Scope", "Eligibility", "Policy", "Procedures", etc.
    
    Returns list of (heading, section_text) tuples.
    If no headings found, returns [("Full Policy", full_text)].
    """
    # Common policy section headers — adapt these to your actual data patterns
    # Look at a few policies manually first to see what patterns exist
    heading_pattern = r'\n\s*(Purpose|Scope|Eligibility|Policy|Procedures?|' \
                      r'Definitions?|Responsibilities|Exceptions?|' \
                      r'Accrual|Usage|Carryover|Request Process|' \
                      r'Reporting|Compliance|Related Policies)\s*\n'
    
    splits = re.split(heading_pattern, text, flags=re.IGNORECASE)
    
    if len(splits) <= 1:
        # No headings found — treat as single section
        return [("Full Policy", text)]
    
    sections = []
    # splits alternates: [preamble, heading1, text1, heading2, text2, ...]
    if splits[0].strip():
        sections.append(("Overview", splits[0].strip()))
    
    for i in range(1, len(splits) - 1, 2):
        heading = splits[i].strip()
        body = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if body:
            sections.append((heading, body))
    
    return sections


def detect_applies_to(text):
    """
    Scan text for role/audience indicators.
    Returns list of role categories this chunk applies to.
    
    This is the CRITICAL function that enables dynamic disambiguation.
    Start with keyword detection, then refine with LLM if needed.
    """
    text_lower = text.lower()
    
    roles_found = set()
    
    # Keyword patterns — customize these based on actual policy language
    patterns = {
        "salaried": [r"salaried\b", r"exempt\b", r"salary\b"],
        "hourly": [r"hourly\b", r"non-exempt\b", r"wage\b"],
        "field": [r"field\b", r"remote\b", r"traveling\b"],
        "store_manager": [r"store manager", r"facility manager", r"club manager"],
        "home_office": [r"home office", r"corporate\b", r"bentonville\b"],
        "management": [r"management\b", r"manager\b", r"supervisor\b", r"lead\b"],
        "part_time": [r"part.time\b", r"part time\b"],
    }
    
    for role, pats in patterns.items():
        for pat in pats:
            if re.search(pat, text_lower):
                roles_found.add(role)
    
    # If nothing specific found, it applies to everyone
    if not roles_found:
        return ["all"]
    
    return list(roles_found)


def chunk_single_policy(policy_id, policy_name, category, state, state_code,
                         text, effective_date, source_url, is_state_variation):
    """
    Process one policy document into parent + child chunks.
    
    Returns (parent_chunks, child_chunks) — both are lists of dicts.
    """
    parent_chunks = []
    child_chunks = []
    
    # Step 1: Split into sections → these become parent chunks
    sections = extract_section_headings(text)
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        length_function=len,  # or use tiktoken for token-accurate splitting
        separators=["\n\n", "\n", ". ", " "]
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    parent_idx = 0
    child_idx = 0
    
    for section_heading, section_text in sections:
        # If section is too long, split it into multiple parents
        if len(section_text) > PARENT_CHUNK_SIZE:
            parent_texts = parent_splitter.split_text(section_text)
        else:
            parent_texts = [section_text]
        
        for pt in parent_texts:
            parent_id = make_chunk_id(policy_id, state_code, parent_idx, "parent")
            
            parent_meta = {
                "policy_id": policy_id,
                "policy_name": policy_name,
                "category": category,
                "state": state,
                "state_code": state_code,
                "is_state_variation": is_state_variation,
                "section_heading": section_heading,
                "effective_date": effective_date,
                "source_url": source_url,
                "applies_to": detect_applies_to(pt),
                "parent_policy_id": policy_id,  # always the base PD-XX
            }
            
            parent_chunks.append({
                "parent_chunk_id": parent_id,
                "text": pt,
                **parent_meta
            })
            
            # Step 2: Split parent into children
            child_texts = child_splitter.split_text(pt)
            
            for ct in child_texts:
                child_id = make_chunk_id(policy_id, state_code, child_idx, "child")
                
                child_chunks.append({
                    "chunk_id": child_id,
                    "parent_chunk_id": parent_id,  # THE LINK
                    "text": ct,
                    # Inherit all parent metadata
                    **parent_meta,
                    "chunk_index": child_idx,
                })
                
                child_idx += 1
            
            parent_idx += 1
    
    return parent_chunks, child_chunks


def process_all_policies():
    """
    Main entry point. Reads all_policies_scraped.json, processes everything.
    """
    with open(SCRAPED_JSON_PATH, "r") as f:
        data = json.load(f)
    
    conn = init_parent_store(PARENT_STORE_PATH)
    all_children = []
    
    for policy in data["policies"]:
        policy_id = policy["policyId"]
        policy_name = policy["name"]
        category = policy["category"]
        
        # --- Process national policy ---
        national_text = policy["content"]["text"]
        if national_text:
            parents, children = chunk_single_policy(
                policy_id=policy_id,
                policy_name=policy_name,
                category=category,
                state=None,
                state_code=None,
                text=national_text,
                effective_date=policy["content"].get("effectiveDate", ""),
                source_url=policy.get("url", ""),
                is_state_variation=False,
            )
            
            # Store parents in SQLite
            for p in parents:
                conn.execute(
                    "INSERT OR REPLACE INTO parent_chunks VALUES (?,?,?,?,?,?,?,?,?)",
                    (p["parent_chunk_id"], p["policy_id"], p["policy_name"],
                     p["state"], p["state_code"], p["category"],
                     p["section_heading"], p["text"], json.dumps(p))
                )
            all_children.extend(children)
        
        # --- Process state variations ---
        for variation in policy.get("stateVariations", []):
            state_text = variation.get("content", {}).get("text", "")
            if not state_text:
                continue
            
            parents, children = chunk_single_policy(
                policy_id=policy_id,
                policy_name=policy_name,
                category=category,
                state=variation["stateName"],
                state_code=variation["stateCode"],
                text=state_text,
                effective_date=variation.get("content", {}).get("effectiveDate", ""),
                source_url=variation.get("url", ""),
                is_state_variation=True,
            )
            
            for p in parents:
                conn.execute(
                    "INSERT OR REPLACE INTO parent_chunks VALUES (?,?,?,?,?,?,?,?,?)",
                    (p["parent_chunk_id"], p["policy_id"], p["policy_name"],
                     p["state"], p["state_code"], p["category"],
                     p["section_heading"], p["text"], json.dumps(p))
                )
            all_children.extend(children)
    
    conn.commit()
    conn.close()
    
    # Write child chunks to JSONL
    output_path = f"{CHUNKS_OUTPUT_DIR}/chunks_enriched.jsonl"
    with open(output_path, "w") as f:
        for child in all_children:
            f.write(json.dumps(child) + "\n")
    
    print(f"Created {len(all_children)} child chunks")
    print(f"Parent chunks stored in {PARENT_STORE_PATH}")
    
    return all_children


if __name__ == "__main__":
    process_all_policies()
```

---

## embed_and_store.py — Hybrid Indexing

```python
"""
embed_and_store.py

Takes chunks_enriched.jsonl → embeds → stores in ChromaDB with metadata.
Also builds a BM25 index for keyword search.

Run this ONCE after chunking (or when policies update).
"""

import json
import pickle
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from config import *


def load_chunks(path):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(texts, client, batch_size=100):
    """
    Embed texts using OpenAI API in batches.
    Returns list of embedding vectors.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def build_chroma_index(chunks, embeddings):
    """
    Store chunks + embeddings + metadata in ChromaDB.
    
    KEY: Store metadata fields as ChromaDB-compatible types.
    ChromaDB metadata supports: str, int, float, bool only.
    Lists must be stored as comma-separated strings (filter with $contains).
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Delete old collection if exists
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except:
        pass
    
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    
    # Prepare for batch insert
    ids = []
    documents = []
    metadatas = []
    embeds = []
    
    for chunk, embedding in zip(chunks, embeddings):
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        embeds.append(embedding)
        
        # Flatten metadata for ChromaDB
        metadatas.append({
            "policy_id": chunk["policy_id"],
            "policy_name": chunk["policy_name"],
            "category": chunk["category"],
            "state": chunk["state_code"] or "national",
            "is_state_variation": chunk["is_state_variation"],
            "section_heading": chunk.get("section_heading", ""),
            "effective_date": chunk.get("effective_date", ""),
            "parent_chunk_id": chunk["parent_chunk_id"],
            "parent_policy_id": chunk.get("parent_policy_id", chunk["policy_id"]),
            # Store list as comma-separated string
            "applies_to": ",".join(chunk.get("applies_to", ["all"])),
        })
    
    # ChromaDB batch insert (max 41666 per batch)
    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            embeddings=embeds[i:i+BATCH],
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )
        print(f"  Stored {min(i+BATCH, len(ids))}/{len(ids)} in ChromaDB")
    
    return collection


def build_bm25_index(chunks):
    """
    Build BM25 index for keyword search.
    Save the index + mapping to disk.
    
    BM25 runs in-memory at query time. For 700 docs / ~5000 chunks,
    this is fast enough (< 10ms per query).
    """
    # Tokenize simply — split on whitespace + lowercase
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    
    # Save BM25 index and chunk mapping
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "chunks": chunks,  # need full chunks for metadata filtering
        }, f)
    
    print(f"BM25 index built over {len(chunks)} chunks")
    return bm25


def main():
    print("Loading chunks...")
    chunks = load_chunks(f"{CHUNKS_OUTPUT_DIR}/chunks_enriched.jsonl")
    
    print(f"Embedding {len(chunks)} chunks...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = embed_texts([c["text"] for c in chunks], openai_client)
    
    print("Building ChromaDB index...")
    build_chroma_index(chunks, embeddings)
    
    print("Building BM25 index...")
    build_bm25_index(chunks)
    
    print("Done! Indexes ready.")


if __name__ == "__main__":
    main()
```

---

## rag_retrieval.py — Hybrid Search + Reranking

This is the core retrieval pipeline: metadata-filtered hybrid search → reranking → parent expansion.

```python
"""
rag_retrieval.py

Three-stage retrieval:
  1. Metadata-filtered hybrid search (dense + BM25)
  2. Cross-encoder reranking
  3. Parent chunk expansion

Returns the final context docs ready for LLM generation.
"""

import json
import pickle
import sqlite3
import numpy as np
from sentence_transformers import CrossEncoder
from openai import OpenAI
import chromadb
from config import *


class RetrievalPipeline:
    def __init__(self):
        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
        
        # BM25
        with open("bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunks = bm25_data["chunks"]
        self.bm25_chunk_ids = bm25_data["chunk_ids"]
        
        # Reranker
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        # Embeddings
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Parent store
        self.parent_conn = sqlite3.connect(PARENT_STORE_PATH)
    
    def embed_query(self, query):
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return response.data[0].embedding
    
    def build_chroma_filters(self, filters):
        """
        Convert our filter dict into ChromaDB where clauses.
        
        filters might look like:
          {"state": "CA", "policy_id": "PD-58", "applies_to": "hourly"}
        
        ChromaDB uses $and, $eq, $contains operators.
        """
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if key == "applies_to":
                # applies_to is stored as "salaried,hourly" — use $contains
                conditions.append({key: {"$contains": value}})
            else:
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
    
    def dense_search(self, query, filters=None, k=INITIAL_RETRIEVE_K):
        """
        Semantic search via ChromaDB with metadata filtering.
        Returns list of (chunk_id, score, metadata, text).
        """
        query_embedding = self.embed_query(query)
        where_clause = self.build_chroma_filters(filters)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
                "source": "dense",
            })
        return output
    
    def bm25_search(self, query, filters=None, k=INITIAL_RETRIEVE_K):
        """
        Keyword search via BM25 with metadata filtering.
        
        BM25 doesn't support metadata natively, so we:
        1. Score ALL chunks with BM25
        2. Filter by metadata post-scoring
        3. Take top-k from the filtered set
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair scores with chunks
        scored = list(zip(scores, self.bm25_chunks))
        
        # Apply metadata filters
        if filters:
            filtered = []
            for score, chunk in scored:
                match = True
                for key, value in filters.items():
                    chunk_val = chunk.get(key) or chunk.get("state_code")
                    if key == "state":
                        chunk_val = chunk.get("state_code") or "national"
                    if key == "applies_to":
                        # Check if value is in the applies_to list
                        if value not in chunk.get("applies_to", ["all"]):
                            match = False
                    elif str(chunk_val) != str(value):
                        match = False
                if match:
                    filtered.append((score, chunk))
            scored = filtered
        
        # Sort by score descending, take top k
        scored.sort(key=lambda x: x[0], reverse=True)
        
        output = []
        for score, chunk in scored[:k]:
            output.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "policy_id": chunk["policy_id"],
                    "policy_name": chunk["policy_name"],
                    "state": chunk.get("state_code") or "national",
                    "parent_chunk_id": chunk["parent_chunk_id"],
                    "applies_to": ",".join(chunk.get("applies_to", ["all"])),
                },
                "score": float(score),
                "source": "bm25",
            })
        return output
    
    def reciprocal_rank_fusion(self, dense_results, bm25_results, k=60):
        """
        Merge two ranked lists using RRF.
        RRF score = sum( 1 / (k + rank) ) across lists.
        No need to normalize scores between different systems.
        """
        scores = {}
        docs = {}
        
        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1 / (k + rank + 1))
            docs[cid] = doc
        
        for rank, doc in enumerate(bm25_results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1 / (k + rank + 1))
            if cid not in docs:
                docs[cid] = doc
        
        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [docs[cid] for cid, score in ranked]
    
    def rerank(self, query, candidates, top_k=RERANK_TOP_K):
        """
        Cross-encoder reranking. Scores each (query, doc) pair jointly.
        Much more accurate than bi-encoder similarity.
        """
        if not candidates:
            return []
        
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
    
    def expand_to_parents(self, reranked_results):
        """
        Replace child chunks with their parent chunks for richer context.
        Deduplicate: if multiple children point to same parent, include parent once.
        """
        seen_parents = set()
        expanded = []
        
        for result in reranked_results:
            parent_id = result["metadata"]["parent_chunk_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            
            # Fetch parent from SQLite
            cursor = self.parent_conn.execute(
                "SELECT text, metadata_json FROM parent_chunks WHERE parent_chunk_id = ?",
                (parent_id,)
            )
            row = cursor.fetchone()
            if row:
                parent_text, parent_meta_json = row
                parent_meta = json.loads(parent_meta_json)
                expanded.append({
                    "text": parent_text,
                    "metadata": parent_meta,
                    "child_rerank_score": result["rerank_score"],
                })
            else:
                # Fallback: use the child chunk itself
                expanded.append(result)
        
        return expanded
    
    def retrieve(self, query, filters=None):
        """
        MAIN ENTRY POINT.
        
        Full pipeline: hybrid search → RRF fusion → rerank → parent expansion.
        
        Args:
            query: user's question (possibly rewritten by agent)
            filters: dict of metadata filters, e.g. {"state": "CA", "applies_to": "hourly"}
        
        Returns:
            list of parent-expanded context documents with metadata
        """
        # Stage 1: Parallel retrieval
        dense_results = self.dense_search(query, filters)
        bm25_results = self.bm25_search(query, filters)
        
        # Stage 2: Fuse
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        
        # Stage 3: Rerank top candidates
        # Take more candidates than final top_k for reranker to work with
        reranked = self.rerank(query, fused[:20])
        
        # Stage 4: Expand to parents
        expanded = self.expand_to_parents(reranked)
        
        # Check confidence
        if expanded and expanded[0].get("child_rerank_score", 0) < RELEVANCE_THRESHOLD:
            # Flag low confidence — agent should caveat its answer
            for doc in expanded:
                doc["low_confidence"] = True
        
        return expanded
    
    def get_policy_variations(self, policy_id):
        """
        Used by the agent to check what jurisdictions/roles a policy varies across.
        Returns metadata about which states and roles have distinct versions.
        
        THIS IS THE KEY TO DYNAMIC DISAMBIGUATION.
        """
        cursor = self.parent_conn.execute(
            """SELECT DISTINCT state_code, 
                      group_concat(DISTINCT json_extract(metadata_json, '$.applies_to'))
               FROM parent_chunks 
               WHERE policy_id = ? 
               GROUP BY state_code""",
            (policy_id,)
        )
        
        variations = {}
        all_roles = set()
        states = set()
        
        for row in cursor:
            state = row[0] or "national"
            states.add(state)
            if row[1]:
                # Parse the applies_to lists
                for role_list in row[1].split(","):
                    for role in role_list.strip("[]'\" ").split(","):
                        role = role.strip("'\" ")
                        if role and role != "all":
                            all_roles.add(role)
        
        return {
            "policy_id": policy_id,
            "has_state_variations": len(states) > 1,
            "states": sorted(states),
            "varies_by_role": len(all_roles) > 0,
            "roles": sorted(all_roles),
        }
    
    def compare_policy(self, policy_id, state_codes):
        """
        Retrieve the full text of a specific policy across multiple jurisdictions.
        Used by the agent for comparison queries.
        """
        results = {}
        for state in state_codes:
            state_filter = state if state != "national" else None
            
            cursor = self.parent_conn.execute(
                """SELECT text, section_heading FROM parent_chunks 
                   WHERE policy_id = ? AND (state_code = ? OR (state_code IS NULL AND ? = 'national'))
                   ORDER BY rowid""",
                (policy_id, state_filter, state)
            )
            
            sections = []
            for row in cursor:
                sections.append({"heading": row[1], "text": row[0]})
            
            results[state] = sections
        
        return results
```

---

## rag_generation.py — Agentic Orchestration

This is the brain. It uses the LLM as a router/planner that decides whether to search, ask for clarification, or compare policies.

```python
"""
rag_generation.py

Agentic RAG orchestration.

The LLM acts as a reasoning agent with access to tools:
  1. search_policies — hybrid retrieval with filters
  2. ask_clarification — return a question to the user
  3. compare_policies — multi-jurisdiction comparison
  4. check_policy_variations — metadata lookup for disambiguation

The agent loop:
  User query → analyze intent → decide action → execute → generate response
  
NO hardcoded conversation trees. The LLM decides dynamically
based on the metadata it discovers.
"""

import json
from openai import OpenAI
from rag_retrieval import RetrievalPipeline
from config import *


# === Tool Definitions (OpenAI function calling format) ===

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": (
                "Search HR policy documents. Use metadata filters to narrow results. "
                "Always apply known filters (state, role type) before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — rephrase the user's question for optimal retrieval"
                    },
                    "state": {
                        "type": "string",
                        "description": "State code filter (e.g. 'CA', 'TX', 'NY') or 'national'. "
                                       "Only apply if you KNOW the user's state."
                    },
                    "applies_to": {
                        "type": "string",
                        "description": "Role filter: 'salaried', 'hourly', 'field', 'store_manager', "
                                       "'home_office', 'management', 'part_time'. "
                                       "Only apply if you KNOW the user's role."
                    },
                    "policy_id": {
                        "type": "string",
                        "description": "Specific policy ID (e.g. 'PD-58') if known."
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_policy_variations",
            "description": (
                "Check what jurisdictions and role types a policy varies across. "
                "Use this BEFORE searching when the user's question is about a policy "
                "that might differ by state or role, and you don't know their state/role yet. "
                "This tells you whether you NEED to ask for clarification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "The policy ID to check (e.g. 'PD-58')"
                    },
                    "policy_name_query": {
                        "type": "string",
                        "description": "If you don't know the exact policy_id, provide the policy name/topic "
                                       "and the system will find the closest match."
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": (
                "Ask the user a clarifying question. Use when: "
                "1) check_policy_variations shows the policy varies by state/role AND "
                "   you don't know the user's state/role from conversation history. "
                "2) The question is too vague to search effectively. "
                "Be specific about WHY you're asking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to ask the user. "
                                       "Include context about why you need this info."
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_policies",
            "description": (
                "Compare a specific policy across multiple states/jurisdictions. "
                "Use when the user explicitly asks about differences between states."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "The policy ID to compare"
                    },
                    "states": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of state codes to compare (e.g. ['CA', 'TX', 'national'])"
                    },
                },
                "required": ["policy_id", "states"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are an HR policy assistant for Walmart associates. You help employees 
understand company policies accurately by searching the policy knowledge base.

CRITICAL RULES:
- ONLY answer based on retrieved policy documents. Never invent policy details.
- If retrieved documents don't contain the answer, say "I couldn't find that specific 
  information in our policy documents" and suggest they contact HR directly.
- Always cite which policy (by name and ID) your answer comes from.
- Keep answers clear and concise. Associates want quick, actionable answers.

DISAMBIGUATION WORKFLOW:
When a user asks about a policy (like PTO, rest breaks, attendance, etc.):
1. First, use check_policy_variations to see if the policy varies by state or role.
2. If it DOES vary, and you DON'T know the user's state/role from the conversation:
   → use ask_clarification to ask ONLY for the dimensions that matter.
   → Don't ask about state if the policy is the same nationwide.
   → Don't ask about role if the policy is the same for all roles.
3. Once you have the needed context, search with appropriate filters.
4. If the policy does NOT vary, just search and answer directly.

USER CONTEXT (from conversation so far):
- State: {user_state}
- Role: {user_role}
- Department: {user_department}

When the user provides their state or role in a response, REMEMBER it for future questions.
"""


class AgentOrchestrator:
    def __init__(self):
        self.retriever = RetrievalPipeline()
        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
    
    def run_agent(self, user_message, conversation_history, user_context):
        """
        Main agent loop.
        
        Args:
            user_message: the user's latest message
            conversation_history: list of {"role": ..., "content": ...} dicts
            user_context: {"state": ..., "role": ..., "department": ...}
        
        Returns:
            {"response": str, "user_context": updated_context, "sources": [...]}
        """
        # Build system prompt with current user context
        system = SYSTEM_PROMPT.format(
            user_state=user_context.get("state", "unknown"),
            user_role=user_context.get("role", "unknown"),
            user_department=user_context.get("department", "unknown"),
        )
        
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history (sliding window — last 10 turns)
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})
        
        # Agent loop — up to 3 tool calls before generating final response
        sources = []
        
        for step in range(4):
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # let the LLM decide
            )
            
            msg = response.choices[0].message
            messages.append(msg.model_dump())
            
            # If the LLM wants to call a tool
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    result = self._execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        user_context,
                        sources,
                    )
                    
                    # If it's a clarification, return it directly to the user
                    if tool_call.function.name == "ask_clarification":
                        return {
                            "response": result,
                            "user_context": user_context,
                            "sources": [],
                            "needs_input": True,
                        }
                    
                    # Feed tool result back to the LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
            else:
                # LLM produced a final response (no tool calls)
                return {
                    "response": msg.content,
                    "user_context": user_context,
                    "sources": sources,
                    "needs_input": False,
                }
        
        # Fallback if too many tool calls
        return {
            "response": msg.content or "I'm having trouble finding that information. "
                        "Could you try rephrasing your question?",
            "user_context": user_context,
            "sources": sources,
            "needs_input": False,
        }
    
    def _execute_tool(self, tool_name, args, user_context, sources):
        """Route tool calls to the right handler."""
        
        if tool_name == "search_policies":
            filters = {}
            if args.get("state"):
                filters["state"] = args["state"]
            if args.get("applies_to"):
                filters["applies_to"] = args["applies_to"]
            if args.get("policy_id"):
                filters["policy_id"] = args["policy_id"]
            
            results = self.retriever.retrieve(args["query"], filters or None)
            
            # Track sources for citation
            for r in results:
                sources.append({
                    "policy_name": r["metadata"].get("policy_name", "Unknown"),
                    "policy_id": r["metadata"].get("policy_id", ""),
                    "state": r["metadata"].get("state", "national"),
                })
            
            # Format context for the LLM
            context_parts = []
            for i, r in enumerate(results):
                meta = r["metadata"]
                header = f"[Source {i+1}: {meta.get('policy_name', 'Policy')} " \
                         f"({meta.get('policy_id', '')}) — " \
                         f"State: {meta.get('state', 'national')}]"
                context_parts.append(f"{header}\n{r['text']}")
            
            return {
                "retrieved_context": "\n\n---\n\n".join(context_parts),
                "num_results": len(results),
                "low_confidence": any(r.get("low_confidence") for r in results),
            }
        
        elif tool_name == "check_policy_variations":
            # Try to find the policy ID from name if not provided
            policy_id = args.get("policy_id")
            
            if not policy_id and args.get("policy_name_query"):
                # Quick search to find the policy ID
                results = self.retriever.dense_search(
                    args["policy_name_query"], k=3
                )
                if results:
                    policy_id = results[0]["metadata"]["policy_id"]
            
            if policy_id:
                variations = self.retriever.get_policy_variations(policy_id)
                return variations
            else:
                return {"error": "Could not identify the policy. Try searching instead."}
        
        elif tool_name == "ask_clarification":
            # Just pass through — the agent loop handles this
            return args["question"]
        
        elif tool_name == "compare_policies":
            comparison = self.retriever.compare_policy(
                args["policy_id"], args["states"]
            )
            
            # Format for LLM
            parts = []
            for state, sections in comparison.items():
                state_text = "\n".join(
                    f"**{s['heading']}**: {s['text']}" for s in sections
                )
                parts.append(f"=== {state.upper()} ===\n{state_text}")
            
            return {
                "comparison_context": "\n\n---\n\n".join(parts),
                "states_compared": list(comparison.keys()),
            }
        
        return {"error": f"Unknown tool: {tool_name}"}
```

---

## app.py — Flask API with Session Management

```python
"""
app.py

Flask API serving the chatbot.
Manages sessions (conversation history + user context per user).
"""

from flask import Flask, request, jsonify, session
from rag_generation import AgentOrchestrator
import uuid
import os

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# In production, use Redis for session storage.
# For now, in-memory dict keyed by session ID.
sessions = {}

agent = AgentOrchestrator()


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "conversation_history": [],
            "user_context": {
                "state": "unknown",
                "role": "unknown",
                "department": "unknown",
            },
        }
    return sessions[session_id]


def update_user_context(user_context, user_message, agent_response):
    """
    Parse user messages for state/role info and update context.
    Called after each exchange.
    
    This can be as simple as regex or as smart as an LLM call.
    Start simple, upgrade later.
    """
    import re
    msg = user_message.lower()
    
    # State detection (simple version)
    from state_utils import detect_state  # your existing utility
    detected_state = detect_state(msg)
    if detected_state:
        user_context["state"] = detected_state
    
    # Role detection
    role_keywords = {
        "salaried": ["salaried", "salary", "exempt"],
        "hourly": ["hourly", "non-exempt", "wage"],
        "store_manager": ["store manager", "sm"],
        "home_office": ["home office", "corporate", "ho"],
        "field": ["field", "remote"],
        "part_time": ["part time", "part-time"],
    }
    for role, keywords in role_keywords.items():
        if any(kw in msg for kw in keywords):
            user_context["role"] = role
            break
    
    return user_context


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    sess = get_session(session_id)
    
    # Run agent
    result = agent.run_agent(
        user_message=user_message,
        conversation_history=sess["conversation_history"],
        user_context=sess["user_context"],
    )
    
    # Update conversation history
    sess["conversation_history"].append({"role": "user", "content": user_message})
    sess["conversation_history"].append({"role": "assistant", "content": result["response"]})
    
    # Update user context from this exchange
    sess["user_context"] = update_user_context(
        sess["user_context"], user_message, result["response"]
    )
    
    return jsonify({
        "response": result["response"],
        "session_id": session_id,
        "sources": result.get("sources", []),
        "needs_input": result.get("needs_input", False),
        "user_context": sess["user_context"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

---

## requirements.txt

```
flask
openai
chromadb
langchain
langchain-community
langchain-openai
rank-bm25
sentence-transformers
tiktoken
```

---

## Execution Order

```bash
# Step 1: Re-chunk with parent-child strategy + metadata enrichment
python chunk_policies.py

# Step 2: Embed and build indexes
python embed_and_store.py

# Step 3: Run the app
python app.py

# Step 4: Test
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "how does the rest break policy work?"}'
```

---

## How the Key Scenarios Play Out

### Scenario 1: "How does the rest break policy work?"
```
Agent thinking:
  → User is asking about rest breaks. I don't know their state or role.
  → Call check_policy_variations(policy_name_query="rest break")
  → Result: has_state_variations=true, states=["national","CA","CO","WA",...], varies_by_role=true
  → I need state and role. Call ask_clarification.

Agent response:
  "The rest break policy varies by state and also depends on your role type. 
   What state do you work in, and are you an hourly or salaried associate?"

User: "California, hourly"

Agent thinking:
  → Now I know: state=CA, role=hourly
  → Call search_policies(query="rest break", state="CA", applies_to="hourly")
  → Got results from "California Meal & Rest Period Policy PD-XX"
  → Generate answer from retrieved context.
```

### Scenario 2: "How does PTO work for store managers in Texas vs California?"
```
Agent thinking:
  → Comparison query. I have both states and a role.
  → Call search_policies(query="PTO", state="TX", applies_to="store_manager")
  → Call search_policies(query="PTO", state="CA", applies_to="store_manager")
  → (or use compare_policies if I know the policy_id)
  → Synthesize comparison from both result sets.
```

### Scenario 3: "What's the dress code?"
```
Agent thinking:
  → Call check_policy_variations(policy_name_query="dress code")
  → Result: has_state_variations=false, varies_by_role=false
  → No disambiguation needed! Just search directly.
  → Call search_policies(query="dress code")
  → Generate answer.
```

### Scenario 4: Follow-up after Scenario 1: "What about overtime?"
```
Agent thinking:
  → User context already has state=CA, role=hourly
  → Call check_policy_variations(policy_name_query="overtime")
  → It varies by state. But I already know they're in CA.
  → Call search_policies(query="overtime", state="CA", applies_to="hourly")
  → Generate answer. No clarification needed.
```

---

## What to Build Later (Phase 2+)

1. **Contextual chunk enrichment**: Batch LLM job to prepend context summaries to each chunk before embedding. ~$5-15 in API costs for 700 docs. Significant retrieval quality boost.

2. **Semantic caching**: Cache (query_embedding → response) pairs. If a new query's embedding is >0.95 similar to a cached one, return the cached response. Huge cost savings for repeated questions.

3. **Evaluation pipeline**: Build a spreadsheet of 100 test questions with expected answers. Run them through the system weekly. Track retrieval recall and answer accuracy over time.

4. **Guardrails**: Post-processing check that the LLM's response only contains claims present in the retrieved context. Flag/block hallucinated policy details.

5. **Admin interface**: Let HR upload new/updated policies, trigger re-indexing, review chat logs for quality.
