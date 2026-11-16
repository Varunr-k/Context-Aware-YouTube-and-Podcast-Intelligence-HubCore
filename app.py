
import os
import re
import sqlite3
import hashlib
from datetime import datetime, timezone

import requests
import feedparser
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# HUBCORE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HubCore Intelligence Hub",
    page_icon="🎧",
    layout="wide"
)

DB_PATH = "hubcore.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_SECONDS = 900

STOP_WORDS = {
    "i", "want", "to", "learn", "about", "the", "a", "an",
    "and", "or", "for", "of", "in", "on", "with", "my",
    "me", "how", "what", "is", "are", "be", "from",
    "beginner", "beginners", "basic", "basics", "best",
    "good", "some", "please", "can", "could", "would",
    "should", "like", "need", "interested", "interest",
    "right", "now", "currently", "video", "videos",
    "podcast", "podcasts", "tutorial", "tutorials",
    "guide", "guides", "introduction", "intro"
}

# Topic synonyms improve search and topic matching.
TOPIC_SYNONYMS = {
    "parkour": [
        "parkour", "freerunning", "free running",
        "vault", "precision jump", "wall run",
        "kong vault", "parkour training", "parkour tutorial"
    ],
    "football": [
        "football", "soccer", "football training",
        "soccer training", "dribbling"
    ],
    "basketball": [
        "basketball", "basketball training",
        "shooting drills", "dribbling drills"
    ],
    "fitness": [
        "fitness", "workout", "strength training",
        "conditioning", "exercise"
    ]
}


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def initialize_database():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY,
                interests TEXT DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                description TEXT,
                action TEXT,
                created_at TEXT
            )
        """)

        conn.execute("""
            INSERT OR IGNORE INTO profile (id, interests)
            VALUES (1, '')
        """)


def get_saved_interests():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT interests FROM profile WHERE id = 1"
        ).fetchone()

    return row[0] if row else ""


def save_interests(interests):
    with get_connection() as conn:
        conn.execute(
            "UPDATE profile SET interests = ? WHERE id = 1",
            (interests,)
        )


def save_feedback(item, action):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO feedback
            (url, title, description, action, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            item["url"],
            item["title"],
            item["description"][:2000],
            action,
            datetime.now(timezone.utc).isoformat()
        ))


def get_feedback():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT title, description, action
            FROM feedback
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return [
        {
            "title": row[0],
            "description": row[1],
            "action": row[2]
        }
        for row in rows
    ]


def clear_feedback():
    with get_connection() as conn:
        conn.execute("DELETE FROM feedback")


initialize_database()


# =========================================================
# TEXT PROCESSING
# =========================================================

def clean_text(value):
    value = str(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def get_keywords(text):
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def expand_topics(text):
    """
    Expand search topics using known synonyms.
    This is used for understanding user interests.
    """

    lower = str(text).lower()
    expanded = set(get_keywords(text))

    for topic, synonyms in TOPIC_SYNONYMS.items():

        if topic in lower or any(
            synonym in lower for synonym in synonyms
        ):
            expanded.update(
                term.lower() for term in synonyms
            )

            expanded.add(topic)

    return expanded


def extract_content_terms(text):
    """
    Extract terms actually present in content.
    Does not award a content item credit for
    synonyms that do not appear in its text.
    """

    lower = str(text).lower()
    terms = get_keywords(lower)

    for synonyms in TOPIC_SYNONYMS.values():

        for phrase in synonyms:

            if phrase in lower:
                terms.add(phrase)

    return terms


def build_search_queries(current_interest, long_term_interest):

    main = (
        current_interest.strip()
        or long_term_interest.strip()
    )

    if not main:
        return []

    queries = [main]
    lower = main.lower()

    # Topic-specific search expansion.

    if "parkour" in lower or "freerunning" in lower:

        queries.extend([
            "parkour beginner training tutorial",
            "parkour basics vault precision jump freerunning"
        ])

    elif "football" in lower or "soccer" in lower:

        queries.extend([
            "beginner football training drills",
            "football skills tutorial"
        ])

    elif "basketball" in lower:

        queries.extend([
            "beginner basketball training drills",
            "basketball skills tutorial"
        ])

    elif any(term in lower for term in [
        "learn", "beginner", "how to", "tutorial", "training"
    ]):

        queries.append(main + " beginner tutorial")

    # Remove duplicate queries.

    unique_queries = []
    seen = set()

    for query in queries:

        key = query.lower().strip()

        if key and key not in seen:
            seen.add(key)
            unique_queries.append(query)

    return unique_queries[:3]


# =========================================================
# AI SEMANTIC MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    return SentenceTransformer(MODEL_NAME)


def get_semantic_scores(items, interest):

    if not items:
        return [], False

    texts = [
        clean_text(
            item["title"] + " " + item["description"]
        )
        for item in items
    ]

    if not interest.strip():
        return [0.0] * len(items), False

    try:

        model = load_ai_model()

        content_vectors = model.encode(
            texts,
            normalize_embeddings=True
        )

        query_vector = model.encode(
            [interest],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            content_vectors,
            query_vector
        ).flatten()

        return scores.tolist(), True

    except Exception as error:

        st.warning(
            "Semantic AI model unavailable. "
            "Using keyword matching instead. "
            + str(error)
        )

        query_words = expand_topics(interest)

        scores = []

        for text in texts:

            content_words = extract_content_terms(text)

            overlap = len(query_words & content_words)

            score = (
                overlap / max(1, len(query_words))
            )

            scores.append(score)

        return scores, False


# =========================================================
# RELEVANCE FILTER AND PERSONALIZED RANKING
# =========================================================

def calculate_recommendations(
    items,
    current_interest,
    long_term_interest
):

    if not items:
        return []

    current_terms = expand_topics(current_interest)
    history_terms = expand_topics(long_term_interest)

    primary_interest = (
        current_interest.strip()
        or long_term_interest.strip()
    )

    semantic_scores, ai_available = get_semantic_scores(
        items,
        primary_interest
    )

    feedback = get_feedback()

    liked_text = [
        extract_content_terms(
            f["title"] + " " + f["description"]
        )
        for f in feedback
        if f["action"] == "like"
    ]

    disliked_text = [
        extract_content_terms(
            f["title"] + " " + f["description"]
        )
        for f in feedback
        if f["action"] == "dislike"
    ]

    ranked = []

    for index, item in enumerate(items):

        content = clean_text(
            item["title"] + " " + item["description"]
        ).lower()

        # IMPORTANT:
        # Only actual content terms are used here.
        content_terms = extract_content_terms(content)

        current_matches = current_terms & content_terms

        history_matches = history_terms & content_terms

        current_keyword_score = (
            len(current_matches)
            / max(1, len(current_terms))
        )

        history_score = (
            len(history_matches)
            / max(1, len(history_terms))
        )

        semantic_score = float(semantic_scores[index])

        # -------------------------------------------------
        # STRICT RELEVANCE FILTER
        # -------------------------------------------------

        if current_terms:

            # Reject results with no direct topic evidence.
            if not current_matches:
                continue

            # Require direct evidence for the selected topic.
            specific_topics = [
                "parkour",
                "football",
                "soccer",
                "basketball",
                "fitness"
            ]

            selected_topics = [
                topic for topic in specific_topics
                if topic in current_interest.lower()
            ]

            if selected_topics:

                has_specific_match = any(
                    term in content
                    for term in selected_topics
                )

                if not has_specific_match:

                    # For parkour, allow its known synonyms.
                    synonym_match = False

                    for topic in selected_topics:

                        for synonym in TOPIC_SYNONYMS.get(
                            topic, []
                        ):

                            if synonym in content:
                                synonym_match = True
                                break

                    if not synonym_match:
                        continue

        else:

            if not history_matches:
                continue

        # -------------------------------------------------
        # SCORE CALCULATION
        # -------------------------------------------------

        if current_interest.strip():

            combined_score = (
                0.55 * max(0.0, semantic_score)
                + 0.30 * current_keyword_score
                + 0.15 * history_score
            )

        else:

            combined_score = (
                0.65 * max(0.0, semantic_score)
                + 0.35 * history_score
            )

        # -------------------------------------------------
        # FEEDBACK PERSONALIZATION
        # -------------------------------------------------

        feedback_adjustment = 0.0

        for liked in liked_text:

            if liked & content_terms:
                feedback_adjustment += 0.02

        for disliked in disliked_text:

            if disliked & content_terms:
                feedback_adjustment -= 0.04

        combined_score += feedback_adjustment

        # -------------------------------------------------
        # STORE RESULT SCORES
        # -------------------------------------------------

        item["current_score"] = current_keyword_score

        item["history_score"] = history_score

        item["semantic_score"] = semantic_score

        item["keyword_score"] = current_keyword_score

        item["score"] = max(
            0.0,
            min(1.0, combined_score)
        )

        item["matched_keywords"] = sorted(
            current_matches
        )

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked


# =========================================================
# YOUTUBE DATA API SEARCH
# =========================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False
)
def search_youtube(query, api_key, limit=15):

    if not api_key:
        return []

    endpoint = (
        "https://www.googleapis.com/youtube/v3/search"
    )

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": "relevance",
        "maxResults": min(limit, 50),
        "safeSearch": "moderate",
        "key": api_key
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=25
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for entry in data.get("items", []):

        video_id = (
            entry.get("id", {}).get("videoId")
        )

        snippet = entry.get("snippet", {})

        if not video_id:
            continue

        thumbnail = (
            snippet.get("thumbnails", {})
            .get("medium", {})
            .get("url", "")
        )

        results.append({
            "platform": "YouTube",
            "title": clean_text(
                snippet.get("title", "")
            ),
            "creator": snippet.get(
                "channelTitle", ""
            ),
            "description": clean_text(
                snippet.get("description", "")
            ),
            "url": (
                "https://www.youtube.com/watch?v="
                + video_id
            ),
            "thumbnail": thumbnail,
            "published": snippet.get(
                "publishedAt", ""
            )
        })

    return results


# =========================================================
# PODCAST DIRECTORY SEARCH
# =========================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False
)
def search_podcast_directory(query):

    endpoint = "https://itunes.apple.com/search"

    params = {
        "term": query,
        "media": "podcast",
        "entity": "podcast",
        "limit": 8
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=25
    )

    response.raise_for_status()

    return response.json().get("results", [])


# =========================================================
# PODCAST RSS EPISODE SEARCH
# =========================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False
)
def get_podcast_episodes(query):

    podcasts = search_podcast_directory(query)

    episodes = []

    seen_feeds = set()

    for podcast in podcasts:

        feed_url = podcast.get("feedUrl")

        if not feed_url or feed_url in seen_feeds:
            continue

        seen_feeds.add(feed_url)

        try:

            response = requests.get(
                feed_url,
                timeout=20,
                headers={
                    "User-Agent": "HubCore/1.0"
                }
            )

            response.raise_for_status()

            feed = feedparser.parse(
                response.content
            )

            for entry in feed.entries[:30]:

                title = clean_text(
                    entry.get("title", "")
                )

                description = clean_text(
                    entry.get(
                        "summary",
                        entry.get("description", "")
                    )
                )

                link = entry.get("link", "")

                if not link:

                    for enclosure in entry.get(
                        "enclosures", []
                    ):

                        link = enclosure.get(
                            "href", ""
                        )

                        if link:
                            break

                if not link:
                    continue

                episodes.append({
                    "platform": "Podcast",
                    "title": title,
                    "creator": podcast.get(
                        "collectionName",
                        "Podcast"
                    ),
                    "description": description,
                    "url": link,
                    "thumbnail": podcast.get(
                        "artworkUrl600", ""
                    ),
                    "published": entry.get(
                        "published", ""
                    )
                })

        except Exception:
            continue

    return episodes


# =========================================================
# COMBINE YOUTUBE AND PODCAST RESULTS
# =========================================================

def discover_content(
    current_interest,
    long_term_interest,
    platform,
    api_key
):

    queries = build_search_queries(
        current_interest,
        long_term_interest
    )

    all_items = []

    errors = []

    for query in queries:

        # YouTube retrieval.

        if platform in ["All", "YouTube"] and api_key:

            try:

                all_items.extend(
                    search_youtube(
                        query,
                        api_key,
                        15
                    )
                )

            except requests.exceptions.HTTPError:

                errors.append(
                    "YouTube API returned an HTTP error. "
                    "Check your API key, API restrictions "
                    "and API quota."
                )

            except Exception as error:

                errors.append(
                    "YouTube search failed: "
                    + str(error)
                )

        # Podcast retrieval.

        if platform in ["All", "Podcasts"]:

            try:

                all_items.extend(
                    get_podcast_episodes(query)
                )

            except Exception as error:

                errors.append(
                    "Podcast search failed: "
                    + str(error)
                )

    # Remove duplicate URLs.

    unique_items = []

    seen_urls = set()

    for item in all_items:

        url = item.get("url", "")

        if url and url not in seen_urls:

            seen_urls.add(url)

            unique_items.append(item)

    return unique_items, errors


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("🎧 HubCore")

st.subheader(
    "Context-Aware YouTube and Podcast Intelligence Hub"
)

st.write(
    "Discover videos and podcast episodes based on "
    "your current intent and long-term interests."
)

st.info(
    "HubCore searches public YouTube and podcast content. "
    "It does not access private YouTube watch history."
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.title("⚙️ HubCore Settings")

api_key = st.sidebar.text_input(
    "YouTube API Key",
    value=os.getenv("YOUTUBE_API_KEY", ""),
    type="password",
    help="Your key is used locally for YouTube Data API access."
)

current_interest = st.sidebar.text_area(
    "What do you want right now?",
    placeholder=(
        "Example: I want to learn parkour "
        "from beginner level."
    ),
    height=100
)

saved_interests = get_saved_interests()

long_term_interest = st.sidebar.text_area(
    "Your long-term interests",
    value=saved_interests,
    placeholder=(
        "Example: Sports, parkour, fitness."
    ),
    height=100
)

platform = st.sidebar.selectbox(
    "Content platform",
    ["All", "YouTube", "Podcasts"]
)

top_k = st.sidebar.slider(
    "Recommendations to show",
    min_value=1,
    max_value=20,
    value=10
)

st.sidebar.caption(
    "Current intent receives priority. "
    "Long-term interests provide additional personalization."
)

refresh_button = st.sidebar.button(
    "🔄 Refresh content",
    use_container_width=True
)

if refresh_button:

    search_youtube.clear()
    search_podcast_directory.clear()
    get_podcast_episodes.clear()

    st.cache_data.clear()

    st.rerun()


# =========================================================
# MAIN SEARCH FORM
# =========================================================

with st.form("hubcore_search_form"):

    discover_button = st.form_submit_button(
        "🚀 Discover Personalized Content",
        use_container_width=True,
        type="primary"
    )


if discover_button:

    if (
        not current_interest.strip()
        and not long_term_interest.strip()
    ):

        st.warning(
            "Please enter your current interest "
            "or long-term interests."
        )

    elif (
        platform == "YouTube"
        and not api_key
    ):

        st.warning(
            "Enter your YouTube API key in the sidebar "
            "to enable YouTube recommendations."
        )

    else:

        save_interests(long_term_interest)

        st.session_state["hubcore_last_search"] = {
            "current_interest": current_interest,
            "long_term_interest": long_term_interest,
            "platform": platform,
            "top_k": top_k,
            "api_key": api_key
        }


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================

search_data = st.session_state.get(
    "hubcore_last_search"
)

if search_data:

    current = search_data["current_interest"]

    history = search_data["long_term_interest"]

    selected_platform = search_data["platform"]

    number_to_show = search_data["top_k"]

    search_api_key = search_data.get(
        "api_key", ""
    )

    st.divider()

    st.subheader("🔎 Search Summary")

    st.write(
        "**Current intent:**",
        current or "Not provided"
    )

    st.write(
        "**Long-term interests:**",
        history or "Not provided"
    )

    st.write(
        "**Platform:**",
        selected_platform
    )

    if (
        selected_platform == "All"
        and not search_api_key
    ):

        st.warning(
            "No YouTube API key was supplied. "
            "Searching podcasts only."
        )

    with st.spinner(
        "Searching and ranking relevant content..."
    ):

        raw_items, search_errors = discover_content(
            current,
            history,
            selected_platform,
            search_api_key
        )

        ranked_items = calculate_recommendations(
            raw_items,
            current,
            history
        )

    for error in search_errors:
        st.warning(error)

    st.divider()

    st.header(
        "✨ Your Personalized Recommendations"
    )

    if not ranked_items:

        st.warning(
            "No sufficiently relevant content was found. "
            "Try a different search phrase or select another platform."
        )

        st.caption(
            "HubCore filters unrelated results instead "
            "of displaying generic sample recommendations."
        )

    else:

        st.success(
            f"Found {len(ranked_items)} relevant items."
        )

        for index, item in enumerate(
            ranked_items[:number_to_show],
            start=1
        ):

            with st.container(border=True):

                col1, col2 = st.columns([1, 3])

                with col1:

                    if item.get("thumbnail"):

                        st.image(
                            item["thumbnail"],
                            use_container_width=True
                        )

                    else:

                        st.write("🎧")

                with col2:

                    st.subheader(
                        f"{index}. {item['title']}"
                    )

                    st.caption(
                        item["platform"]
                        + " | "
                        + item["creator"]
                    )

                    st.write(
                        item["description"][:700]
                        or "No description available."
                    )

                    st.write(
                        "**Overall ranking score:** "
                        + f"{item['score'] * 100:.1f}/100"
                    )

                    st.progress(
                        item["score"]
                    )

                    st.caption(
                        "Current-interest keyword match: "
                        + f"{item['current_score'] * 100:.1f}%"
                    )

                    st.caption(
                        "Long-term-interest match: "
                        + f"{item['history_score'] * 100:.1f}%"
                    )

                    st.caption(
                        "Semantic similarity: "
                        + f"{max(0, item['semantic_score']) * 100:.1f}%"
                    )

                    if item.get("matched_keywords"):

                        st.caption(
                            "Matching terms: "
                            + ", ".join(
                                item["matched_keywords"][:10]
                            )
                        )

                    st.caption(
                        "Published: "
                        + str(
                            item.get("published")
                            or "Not provided"
                        )
                    )

                    st.link_button(
                        "▶ Watch / Listen",
                        item["url"]
                    )

                    item_key = hashlib.sha1(
                        item["url"].encode()
                    ).hexdigest()[:12]

                    button_col1, button_col2 = st.columns(2)

                    with button_col1:

                        if st.button(
                            "👍 Like",
                            key="like_" + item_key
                        ):

                            save_feedback(item, "like")

                            st.toast(
                                "Your preference has been saved."
                            )

                    with button_col2:

                        if st.button(
                            "👎 Not interested",
                            key="dislike_" + item_key
                        ):

                            save_feedback(item, "dislike")

                            st.toast(
                                "Your feedback has been saved."
                            )


# =========================================================
# SAVED PROFILE AND FEEDBACK
# =========================================================

st.divider()

with st.expander(
    "🧠 Saved profile and feedback"
):

    st.write(
        "**Saved long-term interests:**",
        get_saved_interests() or "None saved yet."
    )

    feedback = get_feedback()

    st.write(
        "**Feedback records:**",
        len(feedback)
    )

    if feedback:

        st.dataframe(
            pd.DataFrame(feedback),
            use_container_width=True
        )

        if st.button("Clear all feedback"):

            clear_feedback()

            st.success(
                "Feedback history cleared."
            )

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HubCore | Context-aware cross-media content discovery"
)

st.caption(
    "Content is retrieved through YouTube Data API, "
    "podcast directory and public RSS feeds. "
    "Results are cached for 15 minutes and refreshed on demand. "
    "Ranking scores are not probabilities."
)
