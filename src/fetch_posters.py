import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "movies_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "movie_posters.csv"
)

load_dotenv(BASE_DIR / ".env")

TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")

if not TMDB_API_TOKEN:
    raise RuntimeError(
        "TMDB_API_TOKEN was not found in .env"
    )


# ============================================================
# TMDB CONFIG
# ============================================================

API_URL = "https://api.themoviedb.org/3/movie"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

HEADERS = {
    "Authorization": f"Bearer {TMDB_API_TOKEN}",
    "accept": "application/json",
}


# ============================================================
# LOAD MOVIES
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"Movies found: {len(df)}")


# ============================================================
# FETCH POSTERS
# ============================================================

poster_rows = []

session = requests.Session()
session.headers.update(HEADERS)

for index, row in df.iterrows():

    movie_id = int(row["id"])
    title = row["title"]

    try:

        response = session.get(
            f"{API_URL}/{movie_id}",
            timeout=15,
        )

        if response.status_code == 200:

            data = response.json()

            poster_path = data.get(
                "poster_path"
            )

            if poster_path:

                poster_url = (
                    IMAGE_BASE_URL
                    + poster_path
                )

            else:

                poster_url = ""

            poster_rows.append({
                "id": movie_id,
                "title": title,
                "poster_url": poster_url,
            })

        elif response.status_code == 404:

            poster_rows.append({
                "id": movie_id,
                "title": title,
                "poster_url": "",
            })

        else:

            print(
                f"⚠️ HTTP {response.status_code} "
                f"for {title}"
            )

            poster_rows.append({
                "id": movie_id,
                "title": title,
                "poster_url": "",
            })

    except requests.RequestException as error:

        print(
            f"⚠️ Request failed for {title}: {error}"
        )

        poster_rows.append({
            "id": movie_id,
            "title": title,
            "poster_url": "",
        })

    # Progress
    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(df)} movies"
        )

    # Small delay to be respectful of the API
    time.sleep(0.05)


# ============================================================
# SAVE
# ============================================================

posters_df = pd.DataFrame(
    poster_rows
)

posters_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("✅ Poster data saved")
print(OUTPUT_PATH)
print()
print(
    "Movies with posters:",
    (posters_df["poster_url"] != "").sum()
)