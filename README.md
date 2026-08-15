# 🎬 Hybrid Movie Recommender System

A machine-learning movie recommendation system that combines **content-based filtering** and **collaborative filtering** to generate personalized movie recommendations.

The project uses **TF-IDF** to understand movie content and **SVD** to learn user–movie preferences. A Streamlit application provides an interactive interface, while the TMDB API enriches recommendations with posters, metadata, descriptions, and trailers.

---

## ✨ Features

- 🎯 Personalized movie recommendations
- 🧠 Hybrid recommendation engine
- 📝 Content-based filtering with TF-IDF
- 👥 Collaborative filtering with SVD
- 🔎 Movie search and exploration
- 🎬 Similar-movie recommendations
- 🖼️ TMDB movie posters
- ⭐ Ratings, genres, release dates, and runtime
- 📖 Movie overviews
- 🎥 Trailer integration
- 📊 Model evaluation with RMSE, MAE, Precision@K, Recall@K, and F1@K
- ⚖️ Hybrid-weight experimentation

---

## 🧠 Recommendation Architecture

```text
                    MovieLens + TMDB
                           │
                           ▼
                  Data Preprocessing
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Content-Based              Collaborative
          Filtering                 Filtering
              │                         │
            TF-IDF                     SVD
              │                         │
              └────────────┬────────────┘
                           ▼
                  Hybrid Recommendation
                           │
                           ▼
                  Ranked Movie Results
                           │
                           ▼
                    Streamlit App
                           │
                           ▼
                       TMDB API
```

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | TF-IDF and evaluation |
| Surprise | SVD collaborative filtering |
| SciPy | Sparse matrices |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| TMDB API | Movie metadata, posters and trailers |
| MovieLens | User ratings dataset |
| Git / GitHub | Version control and portfolio |

---

## 📂 Project Structure

```text
movie-recommender/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── tmdb_5000_movies.csv
│   │   └── MovieLens ratings
│   │
│   └── processed/
│       ├── movies_clean.csv
│       ├── movies_features.csv
│       ├── movies_features.pkl
│       └── train_ratings.csv
│
├── models/
│   ├── tfidf_matrix.npz
│   ├── tfidf_vectorizer.joblib
│   ├── svd_model.joblib
│   ├── similarity_matrix.joblib
│   ├── movie_indices.joblib
│   └── hybrid_config.json
│
├── notebooks/
│   └── model development and evaluation notebooks
│
├── src/
│   └── hybrid.py
│
├── requirements.txt
├── .env
└── README.md
```

> Dataset files and API credentials should not be committed to GitHub when they contain private or sensitive information.

---

## 📊 Dataset

### MovieLens

The MovieLens dataset provides user–movie ratings used to train the collaborative filtering component.

The project successfully processed:

- **610 users**
- **9,724 movies**
- **100,836 ratings**

### TMDB

The TMDB dataset/API provides movie information used for content processing and application enrichment.

The processed TMDB catalog contains:

- **4,803 movies**
- movie titles
- overviews
- genres
- keywords
- cast
- crew
- ratings
- popularity

---

## 🔬 Machine Learning

### 1. Content-Based Filtering

Movie metadata is combined into a textual representation and transformed using **TF-IDF**.

The system calculates similarity between movies based on their textual features.

Conceptually:

```text
Movie metadata
      ↓
Text preprocessing
      ↓
TF-IDF
      ↓
Cosine similarity
      ↓
Similar movies
```

### 2. Collaborative Filtering

The collaborative component uses **SVD** to learn relationships between users and movies from historical ratings.

The model was evaluated on a train/test split.

Results obtained during development:

```text
RMSE = 0.8807
MAE  = 0.6766
```

### 3. Hybrid Recommendation

The final recommendation score combines content and collaborative scores:

```text
Hybrid Score =
    content_weight × content_score
    +
    collaborative_weight × collaborative_score
```

The experiments tested multiple weight combinations.

The best configuration obtained during the project was:

```text
Content-Based       10%
Collaborative       90%
```

---

## 📈 Recommendation Evaluation

The recommendation system was evaluated at **K = 10**, with a relevant interaction defined as a rating of **4.0 or higher**.

Results for the selected hybrid configuration:

| Metric | Score |
|---|---:|
| Precision@10 | 0.0240 |
| Recall@10 | 0.0293 |
| F1@10 | 0.0211 |

These metrics are used to evaluate the quality of the ranked top-10 recommendations rather than only the rating-prediction accuracy.

---

## 🎬 Application

The Streamlit application provides:

### Personalized Recommendations

A user enters a MovieLens user ID and receives recommendations generated by the hybrid model.

### Explore Movies

Users can search the movie catalog and select a movie.

### Similar Movies

The content-based model finds movies with similar textual features.

### Movie Details

TMDB integration provides:

- poster
- rating
- release date
- runtime
- genres
- overview
- trailer when available

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
TMDB_API_TOKEN=your_tmdb_api_token
```

Do **not** commit `.env` to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

From the project root:

```powershell
streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🧪 Model Development

The project was developed progressively through notebooks covering:

1. Data exploration
2. Data preprocessing
3. Content-based recommendation
4. SVD collaborative filtering
5. Hybrid recommendation
6. Model evaluation
7. Hybrid-weight comparison
8. Final application integration

---

## 💡 Example Workflow

```text
User ID
   ↓
User rating history
   ↓
Collaborative score
   +
Movie content similarity
   ↓
Hybrid score
   ↓
Top-10 ranking
   ↓
Movie recommendations
```

For movie exploration:

```text
Movie search
   ↓
Selected movie
   ↓
TF-IDF similarity
   ↓
Similar movies
   ↓
TMDB details
```

---

## 🚀 Future Improvements

Possible next improvements include:

- Better cold-start handling for new users
- More advanced collaborative filtering
- Neural recommendation models
- Better ranking metrics and offline evaluation
- Personalized genre preferences
- Recommendation explanations based on specific features
- User authentication and profiles
- Recommendation history
- Cloud deployment
- Automated model retraining
- A production database instead of static processed files

---

## 📌 Portfolio Highlights

This project demonstrates practical experience with:

- Machine Learning
- Recommendation Systems
- Natural Language Processing
- Collaborative Filtering
- Feature Engineering
- Model Evaluation
- Data Processing
- API Integration
- Python Development
- Streamlit Application Development

It combines an actual ML pipeline with a user-facing application rather than presenting only a notebook model.

---

## 👨‍💻 Author

**Mulumba Kakumba**

Computer Engineering Graduate  
Interested in Machine Learning, Data Analysis, Software Engineering, Systems and Cybersecurity.

---

## ⭐ Project Status

**Working prototype / portfolio project**

The core recommendation pipeline and Streamlit application are functional.
