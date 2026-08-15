import numpy as np
import pandas as pd

from .content_based import build_user_content_scores
from .collaborative import get_collaborative_scores


def min_max_normalize(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


def get_hybrid_scores(
    user_id,
    train_ratings,
    df,
    tfidf_matrix,
    svd_model,
    content_weight=0.1,
    collaborative_weight=0.9
):
    """
    Generate hybrid recommendation scores.

    Default configuration:
    10% Content-Based
    90% Collaborative
    """

    content_scores = build_user_content_scores(
        user_id=user_id,
        train_ratings=train_ratings,
        df=df,
        tfidf_matrix=tfidf_matrix
    )

    collaborative_scores = get_collaborative_scores(
        user_id=user_id,
        df=df,
        svd_model=svd_model
    )

    content_normalized = (
        min_max_normalize(
            pd.Series(content_scores)
        ).to_numpy()
    )

    collaborative_normalized = (
        min_max_normalize(
            pd.Series(collaborative_scores)
        ).to_numpy()
    )

    hybrid_scores = (
        content_weight * content_normalized
        + collaborative_weight * collaborative_normalized
    )

    return pd.DataFrame({
        "movieId": df["id"].astype(int),
        "content_score": content_normalized,
        "collaborative_score": collaborative_normalized,
        "hybrid_score": hybrid_scores
    })