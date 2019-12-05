import pandas as pd
import numpy as np
import similar_movie


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

# load my own ratings
myRatings = movieRatings.loc[999].dropna()
print(myRatings)

# loop through my ratings to identify movies similar to the ones I did not like
for i in range(0, len(myRatings.index)):

    # pick movies that I rated low
    if myRatings[i] <= 2:

        # identify similar movies to the ones I did not like
        unlikelyMovies = similar_movie.findSimilarMovies(myRatings.index[i])
        unlikelyMoviesSimilarity = unlikelyMovies['similarity']

        # loop through unlikely movies to identify ones to drop
        for j in range(0, len(unlikelyMoviesSimilarity)):

            # drop unlikely movies
            if unlikelyMoviesSimilarity[j] >= 0.5:
                movieRatings.drop(unlikelyMoviesSimilarity.index[j], axis=1)
                print('You are not going to like ' + unlikelyMoviesSimilarity.index[j])


# get correlation score between every pair of movies
corrMatrix = movieRatings.corr()
# print(corrMatrix.head())

# drop movie similarities that are based on ratings of less than * users
corrMatrix = movieRatings.corr(method='pearson', min_periods=50)
# corrMatrix = movieRatings.corr(method='kendall', min_periods=50)
# corrMatrix = movieRatings.corr(method='spearman', min_periods=50)
# print(corrMatrix.head())

# loop through my ratings to find similar movies
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("\nAdding similarities for " + myRatings.index[i] + "...")
    # retrieve similar movies to the current one
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims.sort_values(inplace = True, ascending = False)
    print(sims[:3])
    # scale similarity by my rating
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)

# some more processing
simCandidates = simCandidates.groupby(simCandidates.index).sum() # group by
simCandidates.sort_values(inplace = True, ascending = False) # sort
finalSims = simCandidates.drop(myRatings.index) # drop movies that I already rated

print("\n\nRESULT: ")
print(finalSims[:10])