import pandas as pd
import numpy as np


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

# get correlation score between every pair of movies
corrMatrix = movieRatings.corr()
# print(corrMatrix.head())

# drop movie similarities that are based on ratings of less than 100 users
corrMatrix = movieRatings.corr(method='pearson', min_periods=50)
# print(corrMatrix.head())

# load my own ratings
myRatings = movieRatings.loc[999].dropna()
print(myRatings)

# loop through my ratings to find similar movies
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding similarities for " + myRatings.index[i] + "...")
    # retrieve similar movies to the current one
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims.sort_values(inplace = True, ascending = False)
    print(sims)
    # scale similarity by my rating
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)
    
# some more processing
simCandidates = simCandidates.groupby(simCandidates.index).sum() # group by
simCandidates.sort_values(inplace = True, ascending = False) # sort
finalSims = simCandidates.drop(myRatings.index) # drop movies that I already rated

print("\n\nRESULT: ")
print(finalSims)