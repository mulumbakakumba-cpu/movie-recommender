import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def build_user_content_scores(
    user_id,
    train_ratings,
    df,
    tfidf_matrix
):
    """
    Build content-based recommendation scores
    for a user using only training ratings.
    """

    user_ratings = train_ratings[
        train_ratings["userId"] == user_id
    ]

    liked_movies = user_ratings[
        user_ratings["rating"] >= 4.0
    ]

    if liked_movies.empty:
        return np.zeros(len(df))

    movie_to_index = {
        int(movie_id): idx
        for idx, movie_id in enumerate(df["id"])
    }

    profile_rows = []
    weights = []

    for _, row in liked_movies.iterrows():

        tmdb_id = int(row["tmdbId"])

        if tmdb_id not in movie_to_index:
            continue

        idx = movie_to_index[tmdb_id]

        profile_rows.append(
            tfidf_matrix[idx]
        )

        weights.append(
            float(row["rating"])
        )

    if not profile_rows:
        return np.zeros(len(df))

    profile_matrix = np.vstack([
        row.toarray().ravel()
        for row in profile_rows
    ])

    profile = np.average(
        profile_matrix,
        axis=0,
        weights=weights
    )

    profile_norm = np.linalg.norm(profile)

    if profile_norm == 0:
        return np.zeros(len(df))

    profile = profile / profile_norm

    content_scores = (
        tfidf_matrix @ profile
    )

    return np.asarray(content_scores).ravel()