import json
from urllib import response

import chromadb
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from tavily import TavilyClient
import os
import requests
from dotenv import load_dotenv
import datetime

load_dotenv()
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_db():
    client = chromadb.PersistentClient(path="./my_local_db")
    return client.get_or_create_collection(name="music_agent")


def _get_known_artists() -> set:
    """Returns a lowercase set of all artist names already in the library."""
    col = get_db()
    results = col.get(where={"type": {"$eq": "artist_record"}})
    return {m["artist"].lower() for m in results.get("metadatas", [])}

@tool
def get_loved_artists(limit: int = 15) -> str:
    """
    Returns artists the user has explicitly loved (status='loved'), sorted by affinity score.
    Use this FIRST to identify seed artists for recommendations — these are the user's
    strongest preferences and should drive genre selection.
    """
    try:
        col = get_db()
        results = col.get(
            where={"$and": [
                {"type": {"$eq": "artist_record"}},
                {"status": {"$eq": "loved"}}
            ]}
        )
        metas = results["metadatas"]
        if not metas:
            return "No loved artists found."

        sorted_loved = sorted(metas, key=lambda x: (x["affinity_score"], x["total_plays"]), reverse=True)[:limit]

        lines = ["### Your Loved Artists (primary seeds for recommendations):"]
        for s in sorted_loved:
            genres = s.get("genres") or "unknown"
            lines.append(
                f"- {s['artist']} | "
                f"Affinity: {s['affinity_score']}/100 | "
                f"Plays: {s['total_plays']} | "
                f"Genres: {genres}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching loved artists: {str(e)}"


@tool
def get_user_top_listening_stats(limit: int = 10) -> str:
    """
    Retrieves the user's highest-affinity and most-played artists.
    Use this first to establish a Taste Profile before searching online.
    """
    try:
        col = get_db()
        results = col.get(
            where={"$and": [
                {"type": {"$eq": "artist_record"}},
                {"affinity_score": {"$gte": 50}}
            ]}
        )
        metas = results["metadatas"]
        if not metas:
            return "No high-affinity artists found."

        sorted_stats = sorted(
            metas,
            key=lambda x: (x["affinity_score"], x["total_plays"]),
            reverse=True
        )[:limit]

        lines = ["### Your Top Artists:"]
        for s in sorted_stats:
            lines.append(
                f"- {s.get('artist')} | "
                f"Affinity: {s.get('affinity_score')}/100 | "
                f"Plays: {s.get('total_plays')} | "
                f"Status: {s.get('status')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error accessing library stats: {str(e)}"


@tool
def get_genre_favorites(genre: str, limit: int = 5) -> str:
    """
    Finds specific artists within a genre ranked by the user's affinity.
    Useful for narrowing down a seed artist for a specific mood.
    """
    try:
        col = get_db()
        results = col.query(
            query_texts=[f"{genre} music"],
            n_results=20,
            where={"type": {"$eq": "artist_record"}}
        )
        metas = results["metadatas"][0]
        if not metas:
            return f"No artists found for genre: {genre}"

        genre_lower = genre.lower()
        matched = [m for m in metas if genre_lower in m.get("genres", "").lower()]

        if not matched:
            return f"No artists in your library tagged with '{genre}'."

        sorted_res = sorted(matched, key=lambda x: x["affinity_score"], reverse=True)[:limit]

        lines = [f"### Your Top {genre.title()} Artists:"]
        for r in sorted_res:
            lines.append(
                f"- {r['artist']} | "
                f"Affinity: {r['affinity_score']}/100 | "
                f"Genres: {r.get('genres', 'unknown')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error in genre search: {str(e)}"

@tool
def find_similar_underground_artists(artist_name: str) -> str:
    """
    Searches Last.fm for niche artists similar to a given artist.
    Filters by listener count (<200k) to ensure underground recommendations only.
    Artists already in the user's library are automatically excluded.
    """
    known = _get_known_artists()

    similar = requests.get("https://ws.audioscrobbler.com/2.0/", params={
        "method": "artist.getSimilar",
        "artist": artist_name,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": 50,
    }).json().get("similarartists", {}).get("artist", [])

    if not similar:
        return f"No similar artists found for {artist_name}."

    niche = []
    for a in similar:
        if a["name"].lower() in known:
            continue
        info = requests.get("https://ws.audioscrobbler.com/2.0/", params={
            "method": "artist.getInfo",
            "artist": a["name"],
            "api_key": LASTFM_API_KEY,
            "format": "json",
        }).json()
        listeners = int(info.get("artist", {}).get("stats", {}).get("listeners", 999_999))
        if listeners < 200_000:
            niche.append((a["name"], listeners))

    if not niche:
        return f"No underground artists found similar to {artist_name} (outside your library)."

    lines = [f"### Underground artists similar to {artist_name} (not in your library):"]
    for name, listeners in sorted(niche, key=lambda x: x[1], reverse=True):
        lines.append(f"- {name} ({listeners:,} listeners)")
    return "\n".join(lines)

@tool
def find_artists_by_genre_tag(genre_tag: str) -> str:
    """
    Finds underground artists tagged under a specific genre on Last.fm.
    Unlike find_similar_underground_artists, this explores the *whole genre pool*
    rather than just neighbours of one artist — use it to discover artists you'd
    never find by following a single 'similar to' chain.
    Artists already in the user's library are automatically excluded.
    """
    known = _get_known_artists()

    response = requests.get("https://ws.audioscrobbler.com/2.0/", params={
        "method": "tag.getTopArtists",
        "tag": genre_tag,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": 50,
    }).json()

    artists = response.get("topartists", {}).get("artist", [])
    if not artists:
        return f"No artists found for genre tag '{genre_tag}'."

    niche = []
    for a in artists:
        if a["name"].lower() in known:
            continue
        info = requests.get("https://ws.audioscrobbler.com/2.0/", params={
            "method": "artist.getInfo",
            "artist": a["name"],
            "api_key": LASTFM_API_KEY,
            "format": "json",
        }).json()
        listeners = int(info.get("artist", {}).get("stats", {}).get("listeners", 999_999))
        if listeners < 200_000:
            niche.append((a["name"], listeners))

    if not niche:
        return f"No underground artists found in genre '{genre_tag}' outside your library."

    lines = [f"### Underground artists in '{genre_tag}' genre (not in your library):"]
    for name, listeners in sorted(niche, key=lambda x: x[1], reverse=True):
        lines.append(f"- {name} ({listeners:,} listeners)")
    return "\n".join(lines)


@tool
def check_artist_status(artist_name: str) -> str:
    """
    Checks if a discovered artist is already in the library or has been disliked.
    Always run this before recommending an artist to avoid repeats.
    """
    try:
        col = get_db()
        results = col.get(
            where={"$and": [
                {"type": {"$eq": "artist_record"}},
                {"artist": {"$eq": artist_name}}
            ]}
        )
        if not results["metadatas"]:
            return f"'{artist_name}' is not in your library. Safe to recommend."

        m = results["metadatas"][0]
        return (
            f"'{artist_name}' is in your library. "
            f"Affinity: {m.get('affinity_score')}/100 | "
            f"Status: {m.get('status')}"
        )
    except Exception as e:
        return f"Error checking artist status: {str(e)}"
    

"""
Gig Tools
"""

@tool
def search_gig_london(genre: str) -> str:
    """
    Search for live music in London tonight
    """

    tavily = TavilyClient(TAVILY_API_KEY)
    today_str = datetime.datetime.now().strftime("%d %B %Y") # e.g., "13 May 2026"

    # Keep the query broad to catch older posts that mention tonight
    niche_query = f"{genre} gigs London tonight (The Windmill Brixton, Shacklewell Arms, Moth Club, Old Blue Last)"

    results = tavily.search(
        query=niche_query,
        search_depth="advanced", # Recommended for finding specific event details
        max_results=10
    )
    print(results)
    for result in results['results']:
        page_markdown = result.get('raw_content')
    
    return page_markdown or "No results found for live music in London tonight."

@tool
def search_favorite_genres(limit: int) -> str:
    """
    Search for news and updates about a favorite genre
    """
    try:
        col = get_db()
        results = col.get(
            where={"$and": [
                {"type": {"$eq": "artist_record"}},
                {"status": {"$eq": "loved"}}
            ]}
        )
        metas = results["metadatas"]
        if not metas:
            return "No loved artists found."

        sorted_loved = sorted(metas, key=lambda x: (x["affinity_score"], x["total_plays"]), reverse=True)[:limit]

        genre_counts: dict[str, int] = {}
        for s in sorted_loved:
            genre_str = s.get("genres")
            if genre_str and genre_str != "unknown":
                for g in genre_str.split(","):
                    g = g.strip().lower()
                    if g:
                        genre_counts[g] = genre_counts.get(g, 0) + 1

        if not genre_counts:
            return "Your loved artists don't have any genres listed."

        top_genres = sorted(genre_counts, key=lambda g: genre_counts[g], reverse=True)

        lines = ["### Your Favourite Genres (from loved artists):"]
        for genre in top_genres:
            lines.append(f"- {genre} ({genre_counts[genre]} artists)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching favourite genres: {str(e)}"

@tool
def search_gigs_tonight(genre: str) -> str:
    """
    Search for live music gigs in London tonight matching a specific genre.
    """

    if not TAVILY_API_KEY:
        return "Error: Tavily API key is missing."

    tavily = TavilyClient(TAVILY_API_KEY)

    today_str = datetime.datetime.now().strftime("%d %B %Y") 

   
    venues = "Windmill Brixton OR Shacklewell Arms OR Moth Club OR Old Blue Last OR Darkhorse OR George Tavern"
    niche_query = f"{genre} live music gigs London tonight {today_str} ({venues})"

    try:
        results = tavily.search(
            query=niche_query,
            search_depth="advanced", 
            max_results=8
        )
        
        gig_lines = [f"### Live {genre.title()} Gigs in London Tonight ({today_str}):\n"]
        
        for result in results.get('results', []):
            title = result.get('title', 'Live Event')
            url = result.get('url', '#')
            snippet = result.get('content', 'No description available.')
            
            gig_lines.append(f"**[{title}]({url})**\n{snippet}\n")
            
        if len(gig_lines) <= 1:
            return f"No results found for {genre} music in London tonight."
            
        return "\n".join(gig_lines)
        
    except Exception as e:
        return f"Error executing gig search: {str(e)}"


@tool
def search_gigs_favorite_genres(limit: int = 5) -> str:
    """
    Finds the user's highest affinity artists, extracts their genres, 
    and searches Tavily for matching live gigs in London tonight.
    """
    try:
        col = get_db()
        
        # 1. Pull the top records using your working metadata type
        results = col.get(
            where={"type": {"$eq": "artist_record"}}
        )
        metas = results.get("metadatas", [])
        if not metas:
            return "No artist records found in your library."

        # 2. Sort by affinity score to get your actual favorites
        # Safely defaults to 0 if affinity_score is missing
        sorted_loved = sorted(
            metas, 
            key=lambda x: x.get("affinity_score", 0), 
            reverse=True
        )[:limit]

        # 3. Extract genres ranked by frequency across top artists
        genre_counts: dict[str, int] = {}
        for s in sorted_loved:
            genre_str = s.get("genres")
            if genre_str and genre_str != "unknown":
                for g in genre_str.split(","):
                    g = g.strip().lower()
                    if g:
                        genre_counts[g] = genre_counts.get(g, 0) + 1

        if not genre_counts:
            return "Your top artists don't have any genres listed to search for."

        top_genres = sorted(genre_counts, key=lambda g: genre_counts[g], reverse=True)[:3]

        # 4. Create a clean search query for Tavily
        today_str = datetime.datetime.now().strftime("%d %B %Y")
        genres_query = " OR ".join(top_genres)
        niche_query = f"live music gigs London {today_str} {genres_query}"
        
        if not TAVILY_API_KEY:
            return "Error: Tavily API key is not configured."

        tavily = TavilyClient(TAVILY_API_KEY)
        search_results = tavily.search(
            query=niche_query,
            search_depth="advanced",
            max_results=5
        )

        gig_lines = [f"### Gigs Tonight Based on Your Top Genres ({', '.join(top_genres)}):\n"]

        for result in search_results.get('results', []):
            title = result.get('title', 'Live Event')
            url = result.get('url', '#')
            snippet = result.get('content', 'No description available.')
            gig_lines.append(f"**[{title}]({url})**\n{snippet}\n")

        if len(gig_lines) <= 1:
            return f"No gig results found for genres: {genres_query}."

        return "\n".join(gig_lines)

    except Exception as e:
        return f"Error fetching gig recommendations: {str(e)}"
    

