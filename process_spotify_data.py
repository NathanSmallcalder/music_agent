"""
Merge artist_profiles.json + listening_events.json into unified format.
embedding_text = factual (no sentiment), metadata = all structured fields.
"""
import json

# ── Config ──────────────────────────────────────
PROFILES = "artist_profiles.json"
EVENTS   = "listening_events.json"
OUTPUT   = "unified_records.json"

# ── Genre → tone mapping (simplistic, refine as needed) ──
GENRE_TONES = {
    "folk": "melancholic, poetic",
    "singer-songwriter": "introspective, acoustic",
    "acoustic": "warm, organic",
    "punk": "raw, energetic, rebellious",
    "post-punk": "angular, brooding, artful",
    "indie rock": "guitar-driven, melodic",
    "noise rock": "dissonant, abrasive",
    "metal": "heavy, aggressive, intense",
    "jazz": "sophisticated, improvisational",
    "electronic": "synthetic, atmospheric",
    "hip hop": "rhythmic, lyrical, streetwise",
    "pop": "catchy, polished, upbeat",
    "r&b": "smooth, soulful, sensual",
    "country": "storytelling, rustic",
    "blues": "soulful, raw, expressive",
    "ambient": "atmospheric, ethereal, spacious",
    "classical": "orchestral, timeless",
    "funk": "groovy, rhythmic, danceable",
    "soul": "emotive, warm, rich",
    "reggae": "laid-back, rhythmic",
    "shoegaze": "dreamy, wall-of-sound",
    "new wave": "synth-driven, quirky",
    "alternative": "offbeat, eclectic",
    "indie": "lo-fi, independent spirit",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def status_from_affinity(score):
    """Map affinity score to a status label."""
    if score >= 70:
        return "loved"
    elif score >= 40:
        return "liked"
    elif score >= 20:
        return "neutral"
    else:
        return "disliked"


def extract_tones(genres):
    """Pick tone descriptors from genre list."""
    tones = set()
    for genre in (g.lower().strip() for g in genres):
        for key, val in GENRE_TONES.items():
            if key in genre:
                for t in val.split(","):
                    tones.add(t.strip())
    return sorted(tones)


def build_embedding_text(artist, genres, tones):
    """Build neutral embedding text — no sentiment baked in."""
    parts = [f"Artist Profile: {artist}."]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")
    if tones:
        parts.append(f"Tone: {', '.join(tones)}.")
    return " ".join(parts)


def main():
    profiles = load_json(PROFILES)
    events   = load_json(EVENTS)

    # Index profiles by artist name (lowercased)
    profile_index = {}
    for p in profiles:
        key = p["artist"].strip().lower()
        profile_index[key] = p

    # Index events by artist name
    event_index = {}
    for e in events:
        key = e["artist"].strip().lower()
        event_index[key] = e

    # Build unified records
    unified = []
    all_artists = set(profile_index.keys()) | set(event_index.keys())

    for artist_key in all_artists:
        profile = profile_index.get(artist_key, {})
        event   = event_index.get(artist_key, {})

        # Merge fields
        artist     = profile.get("artist") or event.get("artist") or artist_key
        genres     = profile.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",") if g.strip()]
        spotify_id = profile.get("spotify_id") or event.get("spotify_id", "")

        affinity   = event.get("affinity_score", 0)
        plays      = event.get("total_plays", 0)
        recency    = event.get("recency_days", 9999)
        status     = status_from_affinity(affinity) if event else "unplayed"

        tones = extract_tones(genres) if genres else []
        embedding = build_embedding_text(artist, genres, tones)

        record = {
            "embedding_text": embedding,
            "metadata": {
                "type":              "artist_record",
                "artist":            artist,
                "genres":            ", ".join(genres),
                "spotify_id":        spotify_id,
                "affinity_score":    affinity,
                "total_plays":       plays,
                "total_playtime_ms": event.get("total_playtime_ms", 0),
                "total_playtime_hours": event.get("total_playtime_hours", 0.0),
                "recency_days":      recency,
                "status":            status,
            }
        }
        unified.append(record)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    # Stats
    with_status = {}
    for r in unified:
        s = r["metadata"]["status"]
        with_status[s] = with_status.get(s, 0) + 1
    print(f"Unified records: {len(unified)}")
    print(f"Status distribution: {with_status}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
