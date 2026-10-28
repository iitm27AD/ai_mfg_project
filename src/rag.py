"""
rag.py  –  Step 4: RAG layer (Document RAG + Graph RAG + Hybrid)
================================================================
1. DOCUMENT RAG : semantic (vector) search over SOP text, past ECO write-ups and any
                  extra documents in docs/ (quality manuals, new SOPs ...).
                  Embeddings + store = ChromaDB (local, free, no API key).
2. GRAPH RAG    : Cypher retrieval of the structured N-hop neighbourhood of a component.
3. HYBRID       : the graph decides WHICH SOPs / past ECOs / documents are relevant to a
                  component; vector search then runs ONLY inside that filtered set to pull
                  the relevant passages.

Usage:
    python src/rag.py index                                  # (re)build the vector index
    python src/rag.py doc    "aluminum machining tolerances" --k 5
    python src/rag.py doc    "surface finish" --type SOP
    python src/rag.py graph  FRAME-TOP-PLATE --hops 2
    python src/rag.py hybrid FRAME-TOP-PLATE "tolerance risk after material change" --k 5

Re-run `index` after loading new documents (Step 3) or dropping files into docs/.

NOTE: GT_IMPACTS (the answer key) is NEVER traversed. ECO nodes are excluded from the
      structural neighbourhood; past ECOs are brought in separately, as text evidence.
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")

CHROMA_DIR = ROOT / "chroma_db"
DOCS_DIR   = ROOT / "docs"
COLLECTION = "eco_knowledge"

# Relationship types an agent must never traverse for structural retrieval
EXCLUDED_RELS = ["GT_IMPACTS", "MENTIONS", "SIMILAR_TO"]

# Node facts worth showing an LLM, per label
KEY_PROPS = {
    "Component": ["nominal_tolerance_mm", "nominal_mass_g", "prod_defect_rate"],
    "Material":  ["yield_strength_mpa", "machinability_index", "raw_cost_index"],
    "Machine":   ["cpk", "maintenance_status", "age_years"],
    "Process":   ["cycle_mean_min", "power_kw"],
    "Supplier":  ["quality_rating", "qualification_status", "lead_time_days"],
    "SOP":       ["revision", "risk"],
    "QualityRequirement": ["type", "tolerance_mm", "spec"],
}


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE (Document RAG)
# ══════════════════════════════════════════════════════════════════════════════

def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    # default embedding function = local MiniLM model (downloaded on first use)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def chunk_text(text: str, max_chars: int = 800) -> list:
    """Paragraph-aware chunking: merge paragraphs up to ~max_chars."""
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    # hard-split any single huge paragraph
    out = []
    for c in chunks:
        out += [c[i:i + max_chars] for i in range(0, len(c), max_chars)] or [c]
    return out


def build_index(session):
    col = get_collection(reset=True)
    ids, docs, metas = [], [], []

    def add(node_id, doc_type, title, text):
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{node_id}::{i}")
            docs.append(chunk)
            metas.append({"node_id": node_id, "doc_type": doc_type, "title": title or node_id})

    # SOPs from the graph
    for r in session.run("MATCH (s:SOP) RETURN s.id AS id, s.name AS name, s.scope AS scope, "
                         "s.critical_tolerances AS tol, s.revision AS rev, s.risk AS risk"):
        text = (f"SOP {r['id']}: {r['name']}\nRevision: {r['rev']}\nScope: {r['scope']}\n\n"
                f"Critical tolerances: {r['tol']}\nRisk classification: {r['risk']}")
        add(r["id"], "SOP", r["name"], text)

    # Past ECO write-ups from the graph (free text only)
    for r in session.run("MATCH (e:ECO) WHERE e.description IS NOT NULL "
                         "RETURN e.id AS id, e.change_type AS ct, e.description AS d"):
        add(r["id"], "ECO", f"{r['id']} ({r['ct']})", f"ECO {r['id']} — {r['ct']}\n\n{r['d']}")

    # Extra unstructured documents: quality manuals, new SOPs (same IDs as Step 3: DOC-<stem>)
    if DOCS_DIR.exists():
        for p in sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))):
            add(f"DOC-{p.stem}", "DOCUMENT", p.name, p.read_text(encoding="utf-8"))

    print(f"Indexing {len(ids)} chunks into ChromaDB …")
    for i in range(0, len(ids), 500):
        col.upsert(ids=ids[i:i + 500], documents=docs[i:i + 500], metadatas=metas[i:i + 500])
    print(f"Done. Index stored in {CHROMA_DIR}")


def doc_rag(query: str, k: int = 5, where: dict = None, doc_type: str = None) -> list:
    """Semantic search. `where` is a Chroma metadata filter (used by hybrid search)."""
    col = get_collection()
    filt = where
    if doc_type:
        type_f = {"doc_type": doc_type}
        filt = {"$and": [where, type_f]} if where else type_f
    res = col.query(query_texts=[query], n_results=k, where=filt)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"node_id": meta["node_id"], "doc_type": meta["doc_type"],
                     "title": meta["title"], "distance": round(dist, 4), "text": text})
    return hits


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH RAG
# ══════════════════════════════════════════════════════════════════════════════

def _label(node) -> str:
    labels = sorted(set(node.labels) - {"Assembly"})
    return labels[0] if labels else "Node"


def _node_str(node) -> str:
    nid, name = node.get("id"), node.get("name")
    return f"{_label(node)}:{nid}" + (f" ({name})" if name and name != nid else "")


def graph_rag(session, component_id: str, hops: int = 2, limit: int = 300) -> dict:
    """
    Structured neighbourhood of a component, up to `hops` hops.
    Excludes ECO/Document nodes and GT_IMPACTS / MENTIONS / SIMILAR_TO edges.
    Returns nodes, edges, ids per label, and an LLM-ready text rendering.
    """
    hops = max(1, min(int(hops), 3))          # clamp: it is interpolated into the Cypher
    query = f"""
        MATCH p = (c:Component {{id: $cid}})-[*1..{hops}]-(n)
        WHERE none(r IN relationships(p) WHERE type(r) IN $excluded)
          AND none(x IN nodes(p) WHERE x:ECO OR x:Document)
        RETURN p LIMIT $limit
    """
    nodes, edges = {}, {}
    for rec in session.run(query, cid=component_id, excluded=EXCLUDED_RELS, limit=limit):
        path = rec["p"]
        for n in path.nodes:
            nodes[n.element_id] = n
        for r in path.relationships:
            edges[r.element_id] = r

    if not nodes:
        return {"component": component_id, "nodes": [], "edges": [], "ids": {}, "text": "",
                "found": False}

    ids_by_label = {}
    for n in nodes.values():
        ids_by_label.setdefault(_label(n), set()).add(n.get("id"))

    edge_lines = sorted({f"{_node_str(r.start_node)} -[{r.type}]-> {_node_str(r.end_node)}"
                         for r in edges.values()})

    fact_lines = []
    for n in sorted(nodes.values(), key=lambda x: (_label(x), str(x.get("id")))):
        props = [f"{k}={n.get(k)}" for k in KEY_PROPS.get(_label(n), []) if n.get(k) is not None]
        if props:
            fact_lines.append(f"{_node_str(n)}: " + ", ".join(props))

    text = (f"GRAPH NEIGHBOURHOOD of Component {component_id} ({hops} hops, "
            f"{len(nodes)} nodes, {len(edges)} relationships)\n\n"
            f"Relationships:\n" + "\n".join(f"  {l}" for l in edge_lines) +
            "\n\nNode facts:\n" + "\n".join(f"  {l}" for l in fact_lines))

    return {"component": component_id, "found": True,
            "nodes": [{"label": _label(n), **dict(n)} for n in nodes.values()],
            "edges": edge_lines, "ids": {k: sorted(v) for k, v in ids_by_label.items()},
            "text": text}


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def hybrid(session, component_id: str, query: str, hops: int = 2, k: int = 5,
           exclude_eco: str = None) -> dict:
    """
    1. GRAPH  : neighbourhood of the component → which SOPs, which past ECOs (those that
                CHANGED any component in the neighbourhood) and which ingested documents
                (MENTIONS a neighbourhood node) are relevant.
    2. VECTOR : semantic search restricted to exactly those nodes' text.
    `exclude_eco` lets you hide the ECO currently being analysed from its own evidence.
    """
    sub = graph_rag(session, component_id, hops)
    if not sub["found"]:
        return {"graph": sub, "passages": [], "candidates": 0}

    comp_ids = sub["ids"].get("Component", [])
    sop_ids  = sub["ids"].get("SOP", [])
    all_ids  = [i for ids in sub["ids"].values() for i in ids]

    eco_ids = [r["id"] for r in session.run(
        "MATCH (e:ECO)-[:CHANGED]->(c:Component) "
        "WHERE c.id IN $ids AND e.id <> coalesce($ex, '') "
        "RETURN DISTINCT e.id AS id LIMIT 300", ids=comp_ids, ex=exclude_eco)]

    doc_ids = [r["id"] for r in session.run(
        "MATCH (d:Document)-[:MENTIONS]->(n) WHERE n.id IN $ids "
        "RETURN DISTINCT d.id AS id", ids=all_ids)]

    candidates = sorted(set(sop_ids) | set(eco_ids) | set(doc_ids))
    if not candidates:
        return {"graph": sub, "passages": [], "candidates": 0}

    passages = doc_rag(query, k=k, where={"node_id": {"$in": candidates}})
    return {"graph": sub, "passages": passages, "candidates": len(candidates),
            "breakdown": {"sops": len(sop_ids), "past_ecos": len(eco_ids),
                          "documents": len(doc_ids)}}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def print_hits(hits: list):
    if not hits:
        print("No matching passages.")
    for i, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")[:300]
        print(f"\n[{i}] {h['doc_type']} · {h['node_id']} · {h['title']}  (distance {h['distance']})")
        print(f"    {snippet}{'…' if len(h['text']) > 300 else ''}")


def main():
    ap = argparse.ArgumentParser(description="RAG layer: document / graph / hybrid retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="(re)build the vector index")

    d = sub.add_parser("doc", help="document RAG (vector search)")
    d.add_argument("query")
    d.add_argument("--k", type=int, default=5)
    d.add_argument("--type", choices=["SOP", "ECO", "DOCUMENT"], help="restrict document type")

    g = sub.add_parser("graph", help="graph RAG (Cypher neighbourhood)")
    g.add_argument("component_id")
    g.add_argument("--hops", type=int, default=2)

    h = sub.add_parser("hybrid", help="graph-filtered vector search")
    h.add_argument("component_id")
    h.add_argument("query")
    h.add_argument("--hops", type=int, default=2)
    h.add_argument("--k", type=int, default=5)
    h.add_argument("--exclude-eco", help="ECO id to hide from its own evidence")

    args = ap.parse_args()

    if args.cmd == "doc":                      # no Neo4j needed
        print_hits(doc_rag(args.query, k=args.k, doc_type=args.type))
        return

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        sys.exit(f"[ERROR] Cannot connect to Neo4j: {e}")

    with driver.session() as s:
        if args.cmd == "index":
            build_index(s)
        elif args.cmd == "graph":
            res = graph_rag(s, args.component_id, args.hops)
            print(res["text"] if res["found"] else f"Component {args.component_id} not found.")
        elif args.cmd == "hybrid":
            res = hybrid(s, args.component_id, args.query, args.hops, args.k, args.exclude_eco)
            if not res["graph"]["found"]:
                sys.exit(f"Component {args.component_id} not found.")
            print(res["graph"]["text"])
            print(f"\n── Graph selected {res['candidates']} candidate documents "
                  f"{res.get('breakdown', {})}; vector search inside that set ──")
            print_hits(res["passages"])
    driver.close()


if __name__ == "__main__":
    main()
