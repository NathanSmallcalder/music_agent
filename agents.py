from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import (
    get_loved_artists,
    get_user_top_listening_stats,
    get_genre_favorites,
    find_similar_underground_artists,
    find_artists_by_genre_tag,
    check_artist_status,
    search_gigs_favorite_genres,
    search_gigs_tonight,
)

# ── LLM ─────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-7b-instruct",
    temperature=0.7,
    max_tokens=32768,
    stop=["<|im_end|>", "<|endoftext|>"]
)

# ── Tools ────────────────────────────────────────────
tool_list = [
    get_loved_artists,
    get_user_top_listening_stats,
    get_genre_favorites,
    find_similar_underground_artists,
    find_artists_by_genre_tag,
    check_artist_status,
    search_gigs_favorite_genres,
    search_gigs_tonight,
]

# ── Agent ────────────────────────────────────────────
agent = create_agent(
    model=llm,
    tools=tool_list,
    system_prompt=(
        "You are a personal music recommendation assistant. "
        "You have access to the user's real Spotify listening history. "
        "Always base recommendations on what the data actually shows, not assumptions. "
        "Never recommend artists already in the user's library. "
        "Seed your discovery ONLY from artists with status='loved' or 'liked' — these are the user's "
        "strongest preferences. Liked or neutral artists are context only, not seeds. "
        "Always call find_similar_underground_artists and find_artists_by_genre_tag before making recommendations. "
        "Do not guess artist names yourself. "
        "When asked about gigs or live music: "
        "First call search_gigs_favorite_genres to find events matching the user's top genres. "
        "Then call search_gigs_tonight with the top genre for additional targeted results. "
        "Always present gig results with name, venue, date, and link."
    )
)

# ── Run ──────────────────────────────────────────────
query = (
    "Step 1: Call get_loved_artists to get the artists I love most. "
    "        These are your ONLY allowed seeds — do not use liked or neutral artists as seeds. "
    "Step 2: From the loved list, identify the distinct genres (e.g. shoegaze, post-punk, dream pop). "
    "        Pick one loved artist per genre as a seed — do not pick two seeds from the same genre. "
    "Step 3: For each seed artist, call find_similar_underground_artists. "
    "Step 4: For each distinct genre from step 2, call find_artists_by_genre_tag with that genre tag "
    "        to explore the broader genre pool beyond just 'sounds like X'. "
    "Step 5: Combine all candidates. Call check_artist_status on any artist you are unsure about. "
    "Step 6: Recommend exactly 5 artists, spanning at least 2 of the genres found in step 2. "
    "        For each recommendation state: artist name, genre, listener count, and one sentence on "
    "        why it fits my taste based on what I actually love."
)

query2 = (
    "Step 1: Call search_gigs_favorite_genres to find London gigs tonight matching my top genres. "
    "Step 2: From the genres returned in Step 1, pick the top genre and call search_gigs_tonight "
    "        with that genre for additional targeted results. "
    "Step 3: Combine all results. Present each gig with: name, venue, date, and link. "
    "        Remove duplicates. If no gigs are found, say so clearly."
)

response = agent.invoke({
    "messages": [("human", query2)]
})

print("\n--- Full Agent Loop ---")
for message in response["messages"]:
    print(type(message).__name__, ":", message.content[:200])
    print("---")

print("\n--- Final Recommendations ---")
print(response["messages"][-1].content)