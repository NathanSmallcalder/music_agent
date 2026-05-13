import json
import chromadb

# ── Config ──────────────────────────────────────
DB_PATH      = "./my_local_db"
COLLECTION   = "music_agent"
UNIFIED      = "data/unified_records.json"

# ── Create collection ───────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION)

# ── Ingest unified records ──────────────────────
def ingest_unified(collection, records):
    docs, ids, metas = [], [], []
    seen = set()
    for r in records:
        sid = r["metadata"]["spotify_id"]
        if sid in seen:
            continue
        seen.add(sid)
        docs.append(r["embedding_text"])
        ids.append(f"artist:{sid}")
        metas.append(r["metadata"])
    if docs:
        collection.upsert(documents=docs, ids=ids, metadatas=metas)

# ── Genre search ────────────────────────────────
def search_by_genre(collection, genre, n=10):
    """Semantic search filtered to artists matching an exact genre string."""
    results = collection.query(
        query_texts=[f"{genre} music"],
        n_results=min(n * 20, 500),
        where={"type": "artist_record"},
        include=["metadatas", "distances"],
    )
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    if metas is None or not metas:
        return []

    matched = []
    for dist, meta in zip(dists, metas):
        genres_str = meta.get("genres", "")
        if genre.lower() in genres_str.lower():
            matched.append((meta, dist))
        if len(matched) >= n:
            break
    return matched


def search_by_status(collection, status, n=10):
    """Semantic search filtering by affinity status (loved/liked/neutral/disliked)."""
    results = collection.query(
        query_texts=["music I enjoy"],
        n_results=n,
        where={"$and": [
            {"type": "artist_record"},
            {"status": status},
        ]},
        include=["metadatas", "distances"],
    )
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    if metas is None or not metas:
        return []
    return list(zip(metas, dists))


# ── Query demo ──────────────────────────────────
def demo_query(collection):
    print("=== genre: post-punk ===")
    for i, (meta, dist) in enumerate(search_by_genre(collection, "post-punk", n=5)):
        print(f"  {i+1}. {meta['artist']}  (dist: {dist:.3f})  genres: {meta['genres']}")

    print("\n=== loved artists (affinity >= 70) ===")
    for i, (meta, dist) in enumerate(search_by_status(collection, "loved", n=5)):
        print(f"  {i+1}. {meta['artist']}  affinity: {meta['affinity_score']}  genres: {meta['genres']}")

    print("\n=== semantic: indie rock with guitar ===")
    results = collection.query(
        query_texts=["indie rock with guitar"],
        n_results=5,
        where={"type": "artist_record"},
        include=["metadatas", "distances"],
    )
    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0], results["distances"][0]
    )):
        print(f"  {i+1}. {meta['artist']}  (dist: {dist:.3f})")


# ── Main ────────────────────────────────────────
def main():
    import os, shutil

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH, ignore_errors=True)

    collection = get_collection()

    with open(UNIFIED, "r", encoding="utf-8") as f:
        ingest_unified(collection, json.load(f))
    print(f"Ingested. Collection count: {collection.count()}")

    demo_query(collection)


if __name__ == "__main__":
    main()
