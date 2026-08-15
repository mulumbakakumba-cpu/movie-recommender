import json
import sys
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import load_npz
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.hybrid import get_hybrid_scores


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(59, 130, 246, 0.12),
                transparent 35%
            ),
            radial-gradient(
                circle at top left,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            #080808;
        color: #ffffff;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        max-width: 1400px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Hero */
    .hero {
        padding: 2rem 0 3rem 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-title span {
        color: #22d3ee;
    }

    .hero-description {
        max-width: 750px;
        margin-top: 1rem;
        color: #9ca3af;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    /* Movie cards */
    .movie-card {
        height: 100%;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.035);
        transition: all 0.2s ease;
    }

    .movie-card:hover {
        border-color: rgba(34,211,238,0.35);
        transform: translateY(-3px);
    }

    .rank {
        color: #22d3ee;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .movie-title {
        margin-top: 0.5rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
    }

    .movie-meta {
        margin-top: 0.7rem;
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .score-box {
        margin-top: 1.2rem;
        padding: 0.8rem;
        border-radius: 12px;
        background: rgba(34,211,238,0.06);
        border: 1px solid rgba(34,211,238,0.10);
    }

    .score-label {
        color: #9ca3af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .score-value {
        margin-top: 0.2rem;
        color: #22d3ee;
        font-size: 1.4rem;
        font-weight: 800;
    }

    /* Section */
    .section-title {
        font-size: 1.8rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0c0c0c;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid rgba(34,211,238,0.35);
        background: rgba(34,211,238,0.10);
        color: #67e8f9;
        font-weight: 700;
        padding: 0.7rem 1rem;
    }

    .stButton > button:hover {
        border-color: #22d3ee;
        background: rgba(34,211,238,0.18);
        color: white;
    }

    /* Footer */
    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #6b7280;
        text-align: center;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_movies():
    return pd.read_csv(
        DATA_DIR / "movies_features.csv"
    )


@st.cache_data
def load_train_ratings():
    return pd.read_csv(
        DATA_DIR / "train_ratings.csv"
    )

@st.cache_data
def load_posters():
    return pd.read_csv(
        DATA_DIR / "movie_posters.csv"
    )


@st.cache_resource
def load_models():

    svd_model = joblib.load(
        MODEL_DIR / "svd_model.joblib"
    )

    tfidf_vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    tfidf_matrix = load_npz(
        MODEL_DIR / "tfidf_matrix.npz"
    )

    with open(
        MODEL_DIR / "hybrid_config.json",
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    return (
        svd_model,
        tfidf_vectorizer,
        tfidf_matrix,
        config,
    )


df = load_movies()
train_ratings = load_train_ratings()
posters = load_posters()

(
    svd_model,
    tfidf_vectorizer,
    tfidf_matrix,
    config,
) = load_models()

def get_similar_movies(
    movie_title,
    df,
    tfidf_matrix,
    top_k=10,
):
    """
    Find movies similar to the selected movie
    using TF-IDF cosine similarity.
    """

    matches = df[
        df["title"].str.lower()
        == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    movie_vector = tfidf_matrix[movie_index]

    scores = (
        tfidf_matrix @ movie_vector.T
    ).toarray().flatten()

    similar_indices = scores.argsort()[::-1]

    similar_indices = [
        i
        for i in similar_indices
        if i != movie_index
    ][:top_k]

    recommendations = df.iloc[
        similar_indices
    ].copy()

    recommendations["similarity_score"] = [
        scores[i]
        for i in similar_indices
    ]

    return recommendations

# ============================================================
# TMDB MOVIE DETAILS
# ============================================================

def get_tmdb_details(movie_id):

    token = os.getenv("TMDB_API_TOKEN")

    if not token:
        return None

    url = (
        f"https://api.themoviedb.org/3/movie/"
        f"{int(movie_id)}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }

    params = {
        "append_to_response": "videos"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None

    # ============================================================
# DISPLAY MOVIE DETAILS
# ============================================================

def display_movie_details(movie_id):

    details = get_tmdb_details(movie_id)

    if not details:

        st.error(
            "Unable to load movie details from TMDB."
        )

        return

    st.markdown("---")

    st.markdown(
        f"""
        <div class="section-title">
            🎬 {details.get("title", "Movie")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        poster_path = details.get(
            "poster_path"
        )

        if poster_path:

            st.image(
                f"https://image.tmdb.org/t/p/w500"
                f"{poster_path}",
                use_container_width=True,
            )

    with col2:

        st.markdown(
            f"### {details.get('title', 'Unknown')}"
        )

        st.write(
            f"⭐ **Rating:** "
            f"{details.get('vote_average', 0):.1f}/10"
        )

        st.write(
            f"📅 **Release:** "
            f"{details.get('release_date', 'N/A')}"
        )

        st.write(
            f"⏱️ **Runtime:** "
            f"{details.get('runtime', 'N/A')} minutes"
        )

        genres = details.get(
            "genres",
            []
        )

        genre_names = [
            genre["name"]
            for genre in genres
        ]

        if genre_names:

            st.write(
                "🎭 **Genres:** "
                + ", ".join(genre_names)
            )

        st.markdown("### 📖 Overview")

        st.write(
            details.get(
                "overview",
                "No overview available.",
            )
        )

    # ========================================================
    # TRAILER
    # ========================================================

    videos = details.get(
        "videos",
        {}
    ).get(
        "results",
        []
    )

    trailer = next(
        (
            video
            for video in videos
            if video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
        ),
        None,
    )

    if trailer:

        st.markdown(
            "### 🎥 Official Trailer"
        )

        st.video(
            f"https://www.youtube.com/watch?v="
            f"{trailer['key']}"
        )

    st.markdown("---")


# ============================================================
# ============================================================
# HERO
# ============================================================

st.markdown(
    "### 🤖 AI-POWERED MOVIE RECOMMENDER"
)

st.markdown(
    "# Discover your next **favorite movie.**"
)

st.write(
    "Personalized movie recommendations powered by a hybrid "
    "machine learning system combining content-based filtering "
    "and collaborative filtering."
)



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎬 Movie Recommender")

    st.caption(
        "Personalized recommendations powered by ML"
    )

    st.markdown("---")

    st.markdown("### 👤 User")

    user_id = st.number_input(
        "MovieLens User ID",
        min_value=1,
        value=1,
        step=1,
    )

    st.markdown("### 🎯 Results")

    top_k = st.slider(
        "Number of movies",
        min_value=5,
        max_value=20,
        value=10,
    )

    st.markdown("---")

    st.markdown("### 🧠 Model")

    st.write(
        f"Content-Based: **{config.get('content_weight', 0.1):.0%}**"
    )

    st.write(
        f"Collaborative: **{config.get('collaborative_weight', 0.9):.0%}**"
    )

    st.caption(
        "Weights selected through offline evaluation."
    )

    st.markdown("---")

st.markdown("### 🔎 Explore Movies")

movie_titles = sorted(
    df["title"]
    .dropna()
    .unique()
    .tolist()
)

selected_movie = st.selectbox(
    "Choose a movie",
    movie_titles,
    index=None,
    placeholder="Search for a movie...",
)

# ============================================================
# FIND SIMILAR MOVIES
# ============================================================

if selected_movie:

    if st.button(
        "✨ Find Similar Movies",
        key="find_similar_movies",
        use_container_width=True,
    ):

        similar_movies = get_similar_movies(
            selected_movie,
            df,
            tfidf_matrix,
            top_k=10,
        )

        st.session_state["similar_movies"] = similar_movies
        st.session_state["similar_movie_title"] = selected_movie

        # Clear an older details selection when generating a new list.
        st.session_state.pop(
            "selected_movie_id",
            None,
        )

        st.rerun()


# ============================================================
# DISPLAY SIMILAR MOVIES
# ============================================================

if "similar_movies" in st.session_state:

    similar_movies = st.session_state["similar_movies"]

    similar_title = st.session_state.get(
        "similar_movie_title",
        "Selected movie",
    )

    st.markdown("---")

    st.markdown(
        f"## 🎬 Movies similar to **{similar_title}**"
    )

    if similar_movies.empty:

        st.warning(
            "No similar movies were found."
        )

    else:

        for start in range(
            0,
            len(similar_movies),
            5,
        ):

            cols = st.columns(5)

            for column_index in range(5):

                movie_index = start + column_index

                if movie_index >= len(similar_movies):
                    break

                movie = similar_movies.iloc[movie_index]
                movie_id = int(movie["id"])

                with cols[column_index]:

                    # Poster
                    poster_match = posters[
                        posters["id"] == movie_id
                    ]

                    poster_url = ""

                    if not poster_match.empty:

                        value = poster_match.iloc[0]["poster_url"]

                        if pd.notna(value):
                            poster_url = str(value)

                    if poster_url:

                        st.image(
                            poster_url,
                            use_container_width=True,
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                height:280px;
                                border-radius:16px;
                                background:#151515;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:3rem;
                            ">
                                🎬
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Movie information
                    st.markdown(
                        f"**#{movie_index + 1} — {movie['title']}**"
                    )

                    st.caption(
                        f"⭐ {movie['vote_average']:.1f}"
                        f"  •  "
                        f"🔥 {movie['popularity']:.1f}"
                    )

                    st.caption(
                        f"🧠 Similarity: "
                        f"{movie['similarity_score']:.3f}"
                    )

                    # IMPORTANT:
                    # This button is INSIDE the movie loop.
                    # Every movie gets a unique key.
                    if st.button(
                        "🎬 View details",
                        key=f"similar_details_{movie_id}",
                        use_container_width=True,
                    ):

                        st.session_state[
                            "selected_movie_id"
                        ] = movie_id

                        st.rerun()


# ============================================================
# SELECTED MOVIE DETAILS
# ============================================================

if "selected_movie_id" in st.session_state:

    selected_id = int(
        st.session_state["selected_movie_id"]
    )

    display_movie_details(selected_id)

    if st.button(
        "✕ Close details",
        key="close_movie_details",
        use_container_width=True,
    ):

        st.session_state.pop(
            "selected_movie_id",
            None,
        )

        st.rerun()

# ============================================================
# USER VALIDATION
# ============================================================

user_exists = (
    user_id in train_ratings["userId"].values
)

if user_exists:

    user_history = train_ratings[
        train_ratings["userId"] == user_id
    ]

    liked_count = len(
        user_history[
            user_history["rating"] >= 4.0
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ratings",
            len(user_history)
        )

    with col2:
        st.metric(
            "Liked Movies",
            liked_count
        )

    with col3:
        st.metric(
            "Model",
            "Hybrid"
        )

else:

    st.warning(
        f"User {user_id} is not available in the training dataset."
    )


# ============================================================
# RECOMMEND
# ============================================================

st.markdown("")

if st.button("🎯 Generate Recommendations"):

    if not user_exists:

        st.error(
            "Please enter a valid MovieLens User ID."
        )

    else:

        with st.spinner(
            "Analyzing preferences and generating recommendations..."
        ):

            # Generate hybrid recommendations
            recommendations = get_hybrid_scores(
                user_id=user_id,
                train_ratings=train_ratings,
                df=df,
                tfidf_matrix=tfidf_matrix,
                svd_model=svd_model,
                content_weight=config.get(
                    "content_weight",
                    0.1,
                ),
                collaborative_weight=config.get(
                    "collaborative_weight",
                    0.9,
                ),
            )

            # Sort and select top K
            recommendations = (
                recommendations
                .sort_values(
                    "hybrid_score",
                    ascending=False,
                )
                .head(top_k)
            )

            # Add movie information
            recommendations = recommendations.merge(
                df[
                    [
                        "id",
                        "title",
                        "vote_average",
                        "popularity",
                    ]
                ],
                left_on="movieId",
                right_on="id",
                how="inner",
            )

            # Add poster URLs
            recommendations = recommendations.merge(
                posters[
                    [
                        "id",
                        "poster_url",
                    ]
                ],
                left_on="movieId",
                right_on="id",
                how="left",
                suffixes=("", "_poster"),
            )

        # ====================================================
        # RECOMMENDATION HEADER
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎬 Recommended for you'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Ranked using the best-performing hybrid configuration.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ====================================================
        # MOVIE GRID
        # ====================================================

        for start in range(
            0,
            len(recommendations),
            2,
        ):

            cols = st.columns(2)

            for column_index in range(2):

                movie_index = start + column_index

                if movie_index >= len(
                    recommendations
                ):
                    break

                movie = recommendations.iloc[
                    movie_index
                ]

                with cols[column_index]:

                    # -----------------------------
                    # POSTER
                    # -----------------------------

                    poster_url = movie.get(
                        "poster_url",
                        "",
                    )

                    if pd.isna(poster_url):
                        poster_url = ""

                    if poster_url:

                        st.image(
                            poster_url,
                            use_container_width=True,
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                height:400px;
                                border-radius:16px;
                                background:#151515;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:4rem;
                            ">
                                🎬
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # -----------------------------
                    # MOVIE INFORMATION
                    # -----------------------------

                    st.markdown(
                        f"""
                        <div class="movie-card">

                            <div class="rank">
                                #{movie_index + 1}
                            </div>

                            <div class="movie-title">
                                {movie["title"]}
                            </div>

                            <div class="movie-meta">
                                ⭐ {movie["vote_average"]:.1f}
                                &nbsp;&nbsp;•&nbsp;&nbsp;
                                🔥 {movie["popularity"]:.1f}
                            </div>

                            <div class="score-box">

                                <div class="score-label">
                                    Hybrid Score
                                </div>

                                <div class="score-value">
                                    {movie["hybrid_score"]:.4f}
                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(
                        min(
                            float(
                                movie["hybrid_score"]
                            ),
                            1.0,
                        )
                    )

                    st.caption(
                        f"🧠 Content: "
                        f"{movie['content_score']:.3f}"
                        f"   •   "
                        f"👥 Collaborative: "
                        f"{movie['collaborative_score']:.3f}"
                    )



# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    "## 📊 Model Performance"
)

st.write(
    "Evaluation results from the Movie Recommender System."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "RMSE",
        "0.8807",
    )

with metric2:
    st.metric(
        "MAE",
        "0.6766",
    )

with metric3:
    st.metric(
        "Precision@10",
        "0.0240",
    )

metric4, metric5 = st.columns(2)

with metric4:
    st.metric(
        "Recall@10",
        "0.0293",
    )

with metric5:
    st.metric(
        "F1@10",
        "0.0211",
    )

st.markdown("### 🏆 Best Hybrid Configuration")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.metric(
        "Content-Based Weight",
        "10%",
    )

with config_col2:
    st.metric(
        "Collaborative Weight",
        "90%",
    )

st.info(
    "The best-performing configuration combines "
    "10% content-based filtering with 90% collaborative filtering."
)


# ============================================================
# MODEL INFORMATION
# ========================================================================
st.markdown("---")

with st.expander("🧠 About this recommendation system"):

    st.markdown(
        """
        ### Hybrid recommendation architecture

        This application combines two recommendation approaches:

        **Content-Based Filtering**

        Uses TF-IDF representations of movie metadata to
        identify movies with similar characteristics.

        **Collaborative Filtering**

        Uses an SVD model trained on MovieLens user-rating
        interactions to predict user preferences.

        **Hybrid Model**

        The two signals are combined using the configuration
        selected through offline evaluation.

        **Best configuration**

        - Content-Based: 10%
        - Collaborative: 90%
        - Precision@10: 0.024027
        - Recall@10: 0.029253
        - F1@10: 0.021134
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Movie Recommender System · TF-IDF + SVD ·
        Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)