# 🎬 Movie Recommendation System

A Machine Learning based Movie Recommendation System built using
Python, Pandas, Scikit-learn and Streamlit.

The system recommends movies based on their similarity using
movie information such as:

- Movie overview
- Genres
- Keywords
- Cast
- Director

---

## 🚀 Features

### 🎬 Movie Recommendations

Enter the name of a movie and the system recommends five similar
movies.

### 🔎 Movie Search

Search for movies by title.

### ⭐ Top Rated Movies

Displays the highest-rated movies from the dataset.

### 🎭 Browse by Genre

Browse movies according to genres such as:

- Action
- Adventure
- Animation
- Comedy
- Crime
- Drama
- Fantasy
- Horror
- Romance
- Science Fiction
- Thriller

---

## 🤖 Machine Learning

The recommendation system uses:

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts the
textual movie information into numerical vectors.

### Cosine Similarity

Cosine Similarity measures how similar two movies are based on
their TF-IDF vectors.

### Hybrid Recommendation Score

The final recommendation score combines:

- 80% content similarity
- 20% movie rating

This helps balance movie similarity with overall movie quality.

---

## 🗂️ Dataset

The project uses the TMDB 5000 Movies and TMDB 5000 Credits
datasets.

The datasets contain information about approximately 5,000 movies,
including:

- Titles
- Movie overviews
- Genres
- Keywords
- Cast
- Crew
- Ratings
- Number of votes

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- NumPy
- TF-IDF
- Cosine Similarity

---

## 📁 Project Structure

```text
Movie recommendation system/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── src/
│   ├── main.py
│   └── app.py
│
├── model/
├── output/
├── requirements.txt
└── README.md