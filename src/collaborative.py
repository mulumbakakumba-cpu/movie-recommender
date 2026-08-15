import numpy as np


def get_collaborative_scores(
    user_id,
    df,
    svd_model
):
    """
    Generate SVD prediction scores for
    every movie in the catalog.
    """

    scores = np.array([
        svd_model.predict(
            int(user_id),
            int(movie_id)
        ).est
        for movie_id in df["id"]
    ])

    return scores