import pandas as pd
import ast
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print("Movie Recommendation System")
print("Number of movies:", len(movies))
print("Number of credits:", len(credits))
print(
    "Average movie rating:",
    round(movies["vote_average"].mean(), 2)
)


# ============================================================
# 2. COMBINE DATASETS
# ============================================================

movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)

print("Datasets combined successfully!")


# ============================================================
# 3. SELECT USEFUL COLUMNS
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
]

movies.rename(
    columns={"title_x": "title"},
    inplace=True
)

print("Useful columns selected!")


# ============================================================
# 4. CONVERT GENRES AND KEYWORDS
# ============================================================

def convert(text):

    items = ast.literal_eval(text)

    result = []

    for item in items:
        result.append(item["name"])

    return result


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


# ============================================================
# 5. GET TOP 3 ACTORS
# ============================================================

def convert_cast(text):

    items = ast.literal_eval(text)

    result = []

    for item in items[:3]:
        result.append(item["name"])

    return result


movies["cast"] = movies["cast"].apply(convert_cast)


# ============================================================
# 6. GET DIRECTOR
# ============================================================

def get_director(text):

    items = ast.literal_eval(text)

    for item in items:

        if item["job"] == "Director":
            return item["name"]

    return ""


movies["crew"] = movies["crew"].apply(get_director)

movies.rename(
    columns={"crew": "director"},
    inplace=True
)


# ============================================================
# 7. HANDLE MISSING VALUES
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
# 8. CREATE TAGS
# ============================================================

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["director"] + " " +
    movies["director"]
)


# ============================================================
# 9. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(
    movies["tags"]
)


# ============================================================
# 10. COSINE SIMILARITY
# ============================================================

similarity = cosine_similarity(vectors)


# ============================================================
# 11. RECOMMEND MOVIES
# ============================================================

def recommend(movie):

    matches = movies[
        movies["title"].str.lower().str.contains(
            movie.lower(),
            na=False
        )
    ]

    if matches.empty:
        print("\nMovie not found.")
        return

    # Multiple movies found
    if len(matches) > 1:

        print("\nMultiple movies found:")
        print("-------------------")

        for number, (_, row) in enumerate(
            matches.head(10).iterrows(),
            start=1
        ):
            print(number, "-", row["title"])

        try:
            choice = int(
                input("\nChoose a movie number: ")
            )

            if choice < 1 or choice > min(len(matches), 10):
                print("Invalid selection.")
                return

        except ValueError:
            print("Please enter a number.")
            return

        index = matches.iloc[choice - 1].name

    else:

        index = matches.index[0]


    # Similarity scores
    distances = similarity[index]

    movie_list = []

    for i, score in enumerate(distances):

        if i == index:
            continue

        rating = movies.iloc[i]["vote_average"]

        # Combined score
        final_score = (
            score * 0.80
            + (rating / 10) * 0.20
        )

        movie_list.append(
            (i, final_score)
        )


    # Sort by final score
    movie_list.sort(
        key=lambda x: x[1],
        reverse=True
    )


    print("\nRecommended Movies:")
    print("-------------------")


    count = 0

    for movie_index, score in movie_list:

        movie = movies.iloc[movie_index]

        print(
            f"{count + 1}. "
            f"{movie['title']} "
            f"⭐ {movie['vote_average']}"
        )

        count += 1

        if count == 5:
            break


# ============================================================
# 12. SEARCH MOVIES
# ============================================================

def search_movies(query):

    results = movies[
        movies["title"].str.lower().str.contains(
            query.lower(),
            na=False
        )
    ]

    if results.empty:

        print("\nNo movies found.")
        return

    print("\nMovies Found:")
    print("--------------------------------")

    for number, (_, movie) in enumerate(
        results.head(10).iterrows(),
        start=1
    ):

        genres = ", ".join(movie["genres"])

        print(
            f"\n{number}. {movie['title']}"
        )

        print(
            f"   Rating: {movie['vote_average']}"
        )

        print(
            f"   Genre: {genres}"
        )


# ============================================================
# 13. TOP RATED MOVIES
# ============================================================

def top_rated_movies():

    results = movies.sort_values(
        by=["vote_average", "vote_count"],
        ascending=False
    )


    print("\nTop Rated Movies:")
    print("-------------------")


    for _, movie in results.head(10).iterrows():

        print(
            movie["title"],
            "- Rating:",
            movie["vote_average"]
        )


# ============================================================
# 14. BROWSE BY GENRE
# ============================================================

def browse_by_genre(genre):

    results = movies[
        movies["genres"].apply(
            lambda x: genre.lower()
            in [g.lower() for g in x]
        )
    ]


    if results.empty:

        print(
            "\nNo movies found for this genre."
        )

        return


    print(
        "\n",
        genre.title(),
        "Movies:"
    )

    print("-------------------")


    for _, movie in results.head(10).iterrows():

        print(
            movie["title"],
            "- Rating:",
            movie["vote_average"]
        )

# ============================================================
# 15. MOVIE DETAILS
# ============================================================

def movie_details(movie_name):

    results = movies[
        movies["title"].str.lower().str.contains(
            movie_name.lower(),
            na=False
        )
    ]

    if results.empty:

        print("\nMovie not found.")
        return

    movie = results.iloc[0]

    genres = ", ".join(movie["genres"])

    print("\nMovie Details")
    print("--------------------------------")

    print("Title:", movie["title"])
    print("Rating:", movie["vote_average"])
    print("Votes:", movie["vote_count"])
    print("Genre:", genres)
    print("Director:", movie["director"])

    print("\nOverview:")
    print(movie["overview"])        


# ============================================================
# 15. MAIN MENU
# ============================================================

while True:

    print(
        "\n=================================================="
    )

    print(
        "        MOVIE RECOMMENDATION SYSTEM"
    )

    print(
        "=================================================="
    )


    print("\n1. Recommend Movies")
    print("2. Search Movies")
    print("3. Top Rated Movies")
    print("4. Browse by Genre")
    print("5. Exit")


    choice = input(
        "\nEnter your choice: "
    )


    # --------------------------------------------------------
    # OPTION 1 - RECOMMEND MOVIES
    # --------------------------------------------------------

    if choice == "1":

        movie_name = input(
            "\nEnter a movie: "
        )

        recommend(movie_name)


    # --------------------------------------------------------
    # OPTION 2 - SEARCH MOVIES
    # --------------------------------------------------------

    elif choice == "2":

        query = input(
            "\nEnter movie name to search: "
        )

        search_movies(query)


    # --------------------------------------------------------
    # OPTION 3 - TOP RATED MOVIES
    # --------------------------------------------------------

    elif choice == "3":

        top_rated_movies()


    # --------------------------------------------------------
    # OPTION 4 - BROWSE BY GENRE
    # --------------------------------------------------------

    elif choice == "4":

        genre = input(
            "\nEnter a genre: "
        )

        browse_by_genre(genre)


    # --------------------------------------------------------
    # OPTION 5 - EXIT
    # --------------------------------------------------------

    elif choice == "5":

        print(
            "\nThank you for using "
            "Movie Recommendation System!"
        )

        print("Goodbye!")

        break


    # --------------------------------------------------------
    # INVALID CHOICE
    # --------------------------------------------------------

    else:

        print("\nInvalid choice.")

        print(
            "Please enter a number from 1 to 5."
        )