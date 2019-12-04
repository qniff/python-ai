import pandas as pd
import numpy as np
import sys

# targetMovie = 'Willy Wonka and the Chocolate Factory (1971)'
# targetMovie = 'Terminator 2: Judgment Day (1991)'
# targetMovie = 'Home Alone (1990)'
# targetMovie = 'Aladdin (1992)'
# targetMovie = 'Aladdin and the King of Thieves (1996)' # hm
# targetMovie = 'Alice in Wonderland (1951)'
# targetMovie = 'Pink Floyd - The Wall (1982)'
# targetMovie = 'Star Trek: Generations (1994)'
# targetMovie = 'Lion King, The (1994)'
targetMovie = 'Star Wars (1977)'
targetMovie = 'Cinderella (1950)'


def findSimilarMovies(targetMovie):
    # read ratings table
    r_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

    # read title table
    m_cols = ['movie_id', 'title']
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")
    # print(movies.sort_values(['title'], ascending=True).to_string()) # to list the movies

    # merge tables together
    ratings = pd.merge(movies, ratings)
    # print(ratings.head())

    # construct a user - rating matrix
    movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
    # print(movieRatings.head())

    # extract specific movie
    targetMovieRatings = movieRatings[targetMovie]

    # compute correlation of target movie with other movies
    similarMovies = movieRatings.corrwith(targetMovieRatings)

    # drop empty data
    similarMovies = similarMovies.dropna()

    # display sorted result data
    # print(similarMovies.sort_values(ascending=False))

    # count how many ratings and what is the average rating
    movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
    # print(movieStats.head())

    # drop movies with less than 100 ratings
    popularMovies = movieStats['rating']['size'] >= 100
    # print(movieStats[popularMovies].sort_values([('rating', 'size')], ascending=True)[:15])

    # join new data set with the original set
    df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
    finalDf = df.drop(targetMovie) # drop movies that I already rated
    print(finalDf.sort_values(['similarity'], ascending=False)[:15])
    return finalDf



findSimilarMovies(targetMovie)