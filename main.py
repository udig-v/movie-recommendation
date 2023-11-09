from flask import Flask, render_template, request
import pandas as pd
from math import sqrt

app = Flask(__name__)

movies_df = pd.read_csv("datasets/movie_data.csv")
ratings_df = pd.read_csv("datasets/ratings_data.csv")


def find_top_movies(user_input):
    inputMovies = pd.DataFrame(user_input)

    # adding movieIds of the input movies to the dataframe
    inputId = movies_df[movies_df["title"].isin(inputMovies["title"].tolist())]
    inputMovies = pd.merge(inputId, inputMovies)

    # removing "year" column
    inputMovies = inputMovies.drop(columns="year")

    # finding users who have seen the same movies and their ratings to those movies
    userSubset = ratings_df[ratings_df["movieId"].isin(inputMovies["movieId"].tolist())]

    # creating sub dataframes, one for each user
    userSubsetGroup = userSubset.groupby(["userId"])

    min_common_movies = 5
    filtered_users = [
        user for user, group in userSubsetGroup if len(group) >= min_common_movies
    ]

    filtered_userSubsetGroup = {}
    for user_id, group in userSubsetGroup:
        if user_id in filtered_users:
            filtered_userSubsetGroup[user_id] = group

    sorted_userSubsetGroup = sorted(
        filtered_userSubsetGroup.items(), key=lambda x: len(x[1]), reverse=True
    )

    # choosing a smaller subset to iterate through
    userSubsetGroup = sorted_userSubsetGroup[0:100]

    # calculating user similarity through Pearson Correlation Coefficient
    pearsonCorrelationDict = {}
    for name, group in userSubsetGroup:
        group = group.sort_values(by="movieId")
        inputMovies = inputMovies.sort_values(by="movieId")

        nRatings = len(group)

        temp_df = inputMovies[inputMovies["movieId"].isin(group["movieId"].tolist())]

        tempRatingList = temp_df["rating"].tolist()

        tempGroupList = group["rating"].tolist()

        Sxx = sum([i**2 for i in tempRatingList]) - pow(
            sum(tempRatingList), 2
        ) / float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(
            nRatings
        )
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(
            tempRatingList
        ) * sum(tempGroupList) / float(nRatings)

        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorrelationDict[name] = 0

    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient="index")
    pearsonDF.columns = ["similarityIndex"]
    pearsonDF["userId"] = pearsonDF.index.str[0]
    pearsonDF.index = range(len(pearsonDF))

    # top 50 users that are most similar to the input user
    topUsers = pearsonDF.sort_values(by="similarityIndex", ascending=False)[0:50]

    # getting the movies watched by users from ratings_df
    topUsersRating = topUsers.merge(
        ratings_df, left_on="userId", right_on="userId", how="inner"
    )

    # multiply each movie rating by its similarity index to find recommendation score
    topUsersRating["weightedRating"] = (
        topUsersRating["similarityIndex"] * topUsersRating["rating"]
    )
    tempTopUsersRating = topUsersRating.groupby("movieId").sum()[
        ["similarityIndex", "weightedRating"]
    ]
    tempTopUsersRating.columns = ["sum_similarityIndex", "sum_weightedRating"]
    recommendation_df = pd.DataFrame()
    recommendation_df["weighted average recommendation score"] = (
        tempTopUsersRating["sum_weightedRating"]
        / tempTopUsersRating["sum_similarityIndex"]
    )
    recommendation_df["movieId"] = tempTopUsersRating.index

    # sorting so the most recommended movies are at the top
    recommendation_df = recommendation_df.sort_values(
        by="weighted average recommendation score", ascending=False
    )
    # recommendation_df.head(10)

    # 10 most recommended movies
    top_movies = movies_df.loc[
        movies_df["movieId"].isin(recommendation_df.head(10)["movieId"].tolist())
    ]

    top_movies = top_movies.to_dict(orient="records")

    return top_movies


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommendations", methods=["POST", "GET"])
def recommendations():
    if request.method == "POST":
        user_movies = request.form.getlist("movies[]")
        user_ratings = request.form.getlist("ratings[]")

        user_input = [
            {"title": movie, "rating": float(rating)}
            for movie, rating in zip(user_movies, user_ratings)
        ]

        top_movies = find_top_movies(user_input)

        return render_template("recommendations.html", recommendations=top_movies)

    return render_template("recommendations.html", recommendations=None)


if __name__ == "__main__":
    app.run(debug=True)
