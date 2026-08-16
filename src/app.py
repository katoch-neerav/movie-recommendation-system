import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 Movie Recommender")

    st.write(
        "A Machine Learning based "
        "movie recommendation system."
    )

    st.divider()

    st.subheader("📌 How it works")

    st.write(
        "The system uses movie information "
        "such as genres, keywords, cast, "
        "director and overview."
    )

    st.write(
        "TF-IDF and Cosine Similarity are "
        "used to find similar movies."
    )

    st.divider()

    st.caption(
        "Built using Python, Pandas "
        "and Scikit-learn"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to your favorite movies."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    movies_url = (
        "https://huggingface.co/datasets/"
        "maximus007411/tmdb-5000-movie-data/"
        "resolve/main/tmdb_5000_movies.csv"
    )

    credits_url = (
        "https://huggingface.co/datasets/"
        "maximus007411/tmdb-5000-movie-data/"
        "resolve/main/tmdb_5000_credits.csv"
    )

    movies = pd.read_csv(movies_url)

    credits = pd.read_csv(credits_url)

    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id"
    )

    return movies


movies = load_data()


# ============================================================
# SELECT USEFUL COLUMNS
# ============================================================

movies = movies[
    [
        "title_x",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "vote_average",
        "vote_count"
    ]
].copy()


movies.rename(
    columns={
        "title_x": "title"
    },
    inplace=True
)


# ============================================================
# CONVERT GENRES AND KEYWORDS
# ============================================================

def convert(text):

    try:

        items = ast.literal_eval(text)

        return [
            item["name"]
            for item in items
        ]

    except:

        return []


movies["genres"] = movies["genres"].apply(convert)

movies["keywords"] = movies["keywords"].apply(convert)


# ============================================================
# GET TOP 3 ACTORS
# ============================================================

def convert_cast(text):

    try:

        items = ast.literal_eval(text)

        return [
            item["name"]
            for item in items[:3]
        ]

    except:

        return []


movies["cast"] = movies["cast"].apply(convert_cast)


# ============================================================
# GET DIRECTOR
# ============================================================

def get_director(text):

    try:

        items = ast.literal_eval(text)

        for item in items:

            if item["job"] == "Director":

                return item["name"]

        return ""

    except:

        return ""


movies["director"] = movies["crew"].apply(get_director)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

movies["overview"] = movies["overview"].fillna("")

movies["director"] = movies["director"].fillna("")


movies["genres"] = movies["genres"].apply(
    lambda x: x if isinstance(x, list) else []
)


movies["keywords"] = movies["keywords"].apply(
    lambda x: x if isinstance(x, list) else []
)


movies["cast"] = movies["cast"].apply(
    lambda x: x if isinstance(x, list) else []
)


# ============================================================
# CREATE TAGS
# ============================================================

movies["tags"] = (
    movies["overview"] + " " +

    movies["genres"].apply(
        lambda x: " ".join(x)
    ) + " " +

    movies["genres"].apply(
        lambda x: " ".join(x)
    ) + " " +

    movies["keywords"].apply(
        lambda x: " ".join(x)
    ) + " " +

    movies["cast"].apply(
        lambda x: " ".join(x)
    ) + " " +

    movies["director"] + " " +

    movies["director"]
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


vectors = vectorizer.fit_transform(
    movies["tags"]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity = cosine_similarity(vectors)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(movie_name):

    matches = movies[
        movies["title"].str.lower().str.contains(
            movie_name.lower(),
            na=False
        )
    ]

    if matches.empty:

        return []


    # Select first matching movie

    index = matches.index[0]


    distances = similarity[index]


    movie_list = []


    for i, score in enumerate(distances):

        if i == index:

            continue


        rating = movies.iloc[i]["vote_average"]


        # Hybrid recommendation score

        final_score = (
            score * 0.80
            + (rating / 10) * 0.20
        )


        movie_list.append(
            (i, final_score)
        )


    movie_list.sort(
        key=lambda x: x[1],
        reverse=True
    )


    recommendations = []


    for movie_index, score in movie_list[:5]:

        movie = movies.iloc[movie_index]


        recommendations.append({

            "title": movie["title"],

            "rating": movie["vote_average"],

            "genre": ", ".join(
                movie["genres"]
            )

        })


    return recommendations


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_movies(query):

    results = movies[
        movies["title"].str.lower().str.contains(
            query.lower(),
            na=False
        )
    ]


    return results.head(10)


# ============================================================
# TOP RATED MOVIES
# ============================================================

def top_rated_movies():

    results = movies.sort_values(

        by=[
            "vote_average",
            "vote_count"
        ],

        ascending=False

    )


    return results.head(10)


# ============================================================
# BROWSE BY GENRE
# ============================================================

def browse_by_genre(genre):

    results = movies[
        movies["genres"].apply(

            lambda x:
            genre.lower()
            in
            [g.lower() for g in x]

        )
    ]


    return results.head(10)


# ============================================================
# PROJECT STATISTICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🎬 Movies",
        len(movies)
    )


with col2:

    st.metric(
        "⭐ Average Rating",
        round(
            movies["vote_average"].mean(),
            2
        )
    )


with col3:

    st.metric(
        "🤖 Algorithm",
        "TF-IDF"
    )


# ============================================================
# TABS
# ============================================================

st.divider()


tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Recommend",
    "🔎 Search",
    "⭐ Top Rated",
    "🎭 Genres"
])


# ============================================================
# TAB 1 — RECOMMEND
# ============================================================

with tab1:

    st.subheader(
        "🎬 Find Similar Movies"
    )


    movie_name = st.text_input(

        "Enter a movie name",

        placeholder="Example: Interstellar",

        key="recommend_movie_input"

    )


    if st.button(

        "🎬 Recommend Movies",

        key="recommend_button"

    ):


        if movie_name.strip() == "":

            st.warning(
                "Please enter a movie name."
            )


        else:

            recommendations = recommend(
                movie_name
            )


            if not recommendations:

                st.error(
                    "Movie not found. "
                    "Try another movie."
                )


            else:

                st.subheader(
                    "⭐ Recommended Movies"
                )


                for number, movie in enumerate(

                    recommendations,

                    start=1

                ):


                    with st.container(
                        border=True
                    ):


                        st.markdown(

                            f"### 🎬 {number}. "
                            f"{movie['title']}"

                        )


                        col1, col2 = st.columns(2)


                        with col1:

                            st.write(

                                f"⭐ Rating: "
                                f"{movie['rating']}"

                            )


                        with col2:

                            st.write(

                                f"🎭 Genre: "
                                f"{movie['genre']}"

                            )


# ============================================================
# TAB 2 — SEARCH
# ============================================================

with tab2:

    st.subheader(
        "🔎 Search Movies"
    )


    search_query = st.text_input(

        "Search for a movie",

        placeholder="Example: Matrix",

        key="search_movie_input"

    )


    if st.button(

        "🔎 Search",

        key="search_button"

    ):


        if search_query.strip() == "":

            st.warning(
                "Please enter a movie name."
            )


        else:

            results = search_movies(
                search_query
            )


            if results.empty:

                st.error(
                    "No movies found."
                )


            else:

                st.subheader(
                    "Movies Found"
                )


                for number, (_, movie) in enumerate(

                    results.iterrows(),

                    start=1

                ):


                    with st.container(
                        border=True
                    ):


                        st.markdown(

                            f"### 🎬 {number}. "
                            f"{movie['title']}"

                        )


                        st.write(

                            f"⭐ Rating: "
                            f"{movie['vote_average']}"

                        )


                        st.write(

                            f"🎭 Genre: "
                            f"{', '.join(movie['genres'])}"

                        )


# ============================================================
# TAB 3 — TOP RATED
# ============================================================

with tab3:

    st.subheader(
        "⭐ Top Rated Movies"
    )


    if st.button(

        "⭐ Show Top Rated Movies",

        key="top_rated_button"

    ):


        results = top_rated_movies()


        for number, (_, movie) in enumerate(

            results.iterrows(),

            start=1

        ):


            with st.container(
                border=True
            ):


                st.markdown(

                    f"### {number}. "
                    f"{movie['title']}"

                )


                st.write(

                    f"⭐ Rating: "
                    f"{movie['vote_average']}"

                )


                st.write(

                    f"👥 Votes: "
                    f"{movie['vote_count']}"

                )


# ============================================================
# TAB 4 — BROWSE BY GENRE
# ============================================================

with tab4:

    st.subheader(
        "🎭 Browse Movies by Genre"
    )


    genres = [

        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Drama",
        "Fantasy",
        "Horror",
        "Romance",
        "Science Fiction",
        "Thriller"

    ]


    selected_genre = st.selectbox(

        "Choose a genre",

        genres

    )


    if st.button(

        "🎭 Browse Movies",

        key="genre_button"

    ):


        results = browse_by_genre(
            selected_genre
        )


        if results.empty:

            st.error(
                "No movies found "
                "for this genre."
            )


        else:

            for number, (_, movie) in enumerate(

                results.iterrows(),

                start=1

            ):


                with st.container(
                    border=True
                ):


                    st.markdown(

                        f"### 🎬 {number}. "
                        f"{movie['title']}"

                    )


                    st.write(

                        f"⭐ Rating: "
                        f"{movie['vote_average']}"

                    )


                    st.write(

                        f"🎭 Genre: "
                        f"{', '.join(movie['genres'])}"

                    )
