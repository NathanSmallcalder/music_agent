"""
Re-fetches genres from Spotify API for artists with empty genres, then
patches both artist_profiles.json and unified_records.json in-place,
and re-ingests into ChromaDB via db.py.

Bypasses process_spotify_data.py since listening_events.json is not present.
"""
import json
import os
import subprocess
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

PROFILES_PATH   = "data/artist_profiles.json"
UNIFIED_PATH    = "data/unified_records.json"
BATCH_SIZE      = 50  # Spotify max per request
LASTFM_API_KEY  = os.getenv("LASTFM_API_KEY")
LASTFM_MAX_TAGS = 5   # top N Last.fm tags to use as genres


def fetch_lastfm_genres(artist_name):
    """Fetch top genre tags for an artist from Last.fm."""
    try:
        resp = requests.get("https://ws.audioscrobbler.com/2.0/", params={
            "method": "artist.getTopTags",
            "artist": artist_name,
            "api_key": LASTFM_API_KEY,
            "format": "json",
        }, timeout=10).json()
        tags = resp.get("toptags", {}).get("tag", [])
        return [t["name"].lower() for t in tags[:LASTFM_MAX_TAGS]]
    except Exception:
        return []


def fetch_genres(sp, profiles):
    """Batch-fetch genres from Spotify, then fall back to Last.fm for empties."""
    missing = [
        (i, p) for i, p in enumerate(profiles)
        if not p.get("genres") and p.get("spotify_id")
    ]
    print(f"Artists with empty genres: {len(missing)}")

    genre_map = {}  # spotify_id -> [genres]
    for batch_start in range(0, len(missing), BATCH_SIZE):
        batch = missing[batch_start: batch_start + BATCH_SIZE]
        ids = [p["spotify_id"] for _, p in batch]
        result = sp.artists(ids)
        for artist in result["artists"]:
            if artist and artist.get("genres"):
                genre_map[artist["id"]] = artist["genres"]

    # Last.fm fallback for artists Spotify couldn't classify
    still_missing = [(i, p) for i, p in missing if p["spotify_id"] not in genre_map]
    print(f"Falling back to Last.fm for {len(still_missing)} artists ...")
    for i, p in still_missing:
        tags = fetch_lastfm_genres(p["artist"])
        if tags:
            genre_map[p["spotify_id"]] = tags
            print(f"  Last.fm genres for {p['artist']}: {tags}")

    return missing, genre_map


def patch_profiles(profiles, missing, genre_map):
    updated = 0
    for idx, profile in missing:
        genres = genre_map.get(profile["spotify_id"])
        if not genres:
            continue
        profiles[idx]["genres"] = genres
        name = profile["artist"]
        pop = profile.get("popularity", 0)
        followers = profile.get("followers", 0)
        genre_str = ", ".join(genres)
        similar = profile.get("similar_artists", [])
        similar_str = (", ".join(similar) + ". ") if similar else ""
        profiles[idx]["description"] = (
            f"{name} is an artist with a popularity score of {pop}/100 on Spotify. "
            f"They have {followers:,} followers and their music spans genres such as {genre_str}."
        )
        profiles[idx]["embedding_text"] = (
            f"{name}. Genres: {genre_str}. "
            f"Similar to: {similar_str}"
            f"Popularity {pop}/100. {followers:,} followers."
        )
        print(f"  Patched {name}: {genres}")
        updated += 1
    return updated


def patch_unified(unified, genre_map_by_name):
    """Update genres and embedding_text in unified_records for patched artists."""
    updated = 0
    for record in unified:
        meta = record.get("metadata", {})
        artist_name = meta.get("artist", "")
        genres = genre_map_by_name.get(artist_name.lower())
        if not genres:
            continue
        genre_str = ", ".join(genres)
        meta["genres"] = genre_str
        record["metadata"] = meta
        # Rebuild embedding text
        tones_part = ""
        record["embedding_text"] = (
            f"Artist Profile: {artist_name}. "
            f"Genres: {genre_str}."
        )
        print(f"  Unified patched: {artist_name}")
        updated += 1
    return updated


def main():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    ))

    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    missing, genre_map = fetch_genres(sp, profiles)

    if not genre_map:
        print("No new genres found from Spotify. Nothing to update.")
        return

    # Build a name->genres map for patching unified_records
    id_to_name = {p["spotify_id"]: p["artist"] for _, p in missing}
    genre_map_by_name = {
        id_to_name[sid].lower(): genres
        for sid, genres in genre_map.items()
        if sid in id_to_name
    }

    # Patch artist_profiles.json
    n_profiles = patch_profiles(profiles, missing, genre_map)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {PROFILES_PATH} ({n_profiles} artists updated)")

    # Patch unified_records.json
    with open(UNIFIED_PATH, "r", encoding="utf-8") as f:
        unified = json.load(f)
    n_unified = patch_unified(unified, genre_map_by_name)
    with open(UNIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"Saved {UNIFIED_PATH} ({n_unified} records updated)")

    # Re-ingest into ChromaDB
    print("\nRe-running db.py ...")
    subprocess.run(["python", "db.py"], check=True)
    print("\nDone. ChromaDB is up to date.")


if __name__ == "__main__":
    main()
