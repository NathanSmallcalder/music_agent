#  Spotify Music Discovery Agent

A local AI agent that analyses your real Spotify streaming history (2015–2026) to recommend underground artists and surface live gigs in London tonight — all grounded in what you actually listen to, never guesswork.

---

## How It Works

```
Spotify History JSONs
        │
        ▼
process_spotify_data.py   ← merges artist profiles + listening events
        │
        ▼
  unified_records.json    ← structured artist records with affinity scores
        │
patch_missing_genres.py   ← backfills empty genres (Spotify API → Last.fm)
        │
        ▼
      db.py               ← ingests records into ChromaDB (vector store)
        │
        ▼
    my_local_db/          ← persistent ChromaDB collection
        │
        ▼
     tools.py             ← LangChain tools querying the DB + external APIs
        │
        ▼
     agents.py            ← LM Studio LLM agent that orchestrates the tools
```

---

## Features

### Underground Artist Discovery
- Calls **Last.fm** to find artists similar to your most-loved seeds
- Filters to artists with **< 200 000 listeners** to keep recommendations truly niche
- Separately explores the **full genre pool** via `tag.getTopArtists` — not just the "sounds like X" chain
- Automatically excludes artists already in your library

###  Affinity Scoring
Your listening history is condensed into an affinity score per artist:

| Score | Status   | Used as         |
|-------|----------|-----------------|
| ≥ 70  | `loved`  | Primary seeds   |
| ≥ 40  | `liked`  | Context only    |
| ≥ 20  | `neutral`| Context only    |
| < 20  | `disliked` | Excluded      |

### London Gig Finder
- Extracts your top genres from loved artists
- Searches **Tavily** for live music tonight at niche London venues
  (Windmill Brixton, Shacklewell Arms, Moth Club, Old Blue Last, George Tavern…)
- Two-step search: broad genre sweep → targeted per-genre follow-up

###  Local LLM
Runs entirely on your machine via **LM Studio** (default model: `qwen2.5-7b-instruct-uncensored`). No cloud inference needed for the agent itself.

---

## Project Structure

```
my_spotify_data/
├── Streams/                         # Raw Spotify extended history JSONs (gitignored)
├── data/
│   ├── artist_profiles.json         # Spotify artist metadata (genres, popularity, followers)
│   ├── unified_records.json         # Merged + scored records ready for ingestion
│   └── spotify_data_*.json          # Intermediate export snapshots
├── my_local_db/                     # ChromaDB vector store (gitignored)
├── process_spotify_data.py          # Merge artist profiles + listening events
├── patch_missing_genres.py          # Backfill genres via Spotify → Last.fm
├── db.py                            # Ingest unified records into ChromaDB
├── tools.py                         # LangChain tools (library queries + external APIs)
├── agents.py                        # Agent definition + query runner
├── .env                             # API credentials (see below, never commit)
└── .gitignore
```

---

## Setup

### 1. Prerequisites

- Python 3.12+
- [LM Studio](https://lmstudio.ai/) running locally on `http://localhost:1234`
  - Load any instruction-tuned model (tested with `qwen2.5-7b-instruct`)

### 2. Install dependencies

```bash
pip install langchain langchain-openai langchain-community chromadb spotipy tavily-python python-dotenv requests
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=https://example.com/callback
LASTFM_API_KEY=your_lastfm_api_key
TAVILY_API_KEY=your_tavily_api_key
```

- **Spotify**: [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard/) → Create app
- **Last.fm**: [last.fm/api/account/create](https://www.last.fm/api/account/create)
- **Tavily**: [tavily.com](https://tavily.com) → free tier available

### 4. Add your Spotify data

Export your **Extended Streaming History** from Spotify (Account → Privacy → Download your data → Extended streaming history). Place the resulting `Streaming_History_Audio_*.json` files in the `Streams/` folder.

### 5. Process your data

```bash
# Step 1: merge profiles + listening events into unified records
python process_spotify_data.py

# Step 2: backfill any missing genres (Spotify API → Last.fm fallback)
python patch_missing_genres.py

# Step 3: ingest into ChromaDB
python db.py
```

> `patch_missing_genres.py` already calls `db.py` at the end, so steps 2 & 3 can be combined.

---

## Running the Agent

Edit the bottom of `agents.py` to choose which query to run:

```python
# Music discovery (underground artist recommendations)
response = agent.invoke({"messages": [("human", query)]})

# Gig finder (London live music tonight)
response = agent.invoke({"messages": [("human", query2)]})
```

Then run:

```bash
python agents.py
```

The agent prints each tool call and its final recommendations to stdout.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `get_loved_artists` | Returns your top `loved` artists by affinity score — primary seeds |
| `get_user_top_listening_stats` | Returns all artists with affinity ≥ 50 |
| `get_genre_favorites` | Finds your top artists within a specific genre |
| `find_similar_underground_artists` | Last.fm similar artists filtered to < 200k listeners |
| `find_artists_by_genre_tag` | Last.fm genre tag pool filtered to < 200k listeners |
| `check_artist_status` | Checks if an artist is already in your library |
| `search_gigs_tonight` | Tavily search for a specific genre's gigs in London tonight |
| `search_gigs_favorite_genres` | Combines genre extraction + Tavily gig search automatically |

---

## Agent Prompts

### Discovery (`query`)
Runs a structured 6-step workflow:
1. Fetches loved artists as seeds
2. Identifies distinct genres, one seed per genre
3. Finds underground similar artists per seed
4. Explores the full genre pool via tags
5. Deduplicates and checks library status
6. Recommends exactly 5 artists across ≥ 2 genres with reasoning

### Gig Finder (`query2`)
1. Calls `search_gigs_favorite_genres` for a broad genre sweep
2. Picks the top genre and calls `search_gigs_tonight` for targeted results
3. Combines, deduplicates, and presents all gigs with name, venue, date, and link

---
