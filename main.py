import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

#Importing Dataset
ratings_data = pd.read_csv('ratings.csv')
movie_data = pd.read_csv('movies.csv')

#Processing data
movie_ratings_data = pd.merge(ratings_data, movie_data, on='movieId')
user_movie_matrix = movie_ratings_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
train_data, test_data = train_test_split(user_movie_matrix, test_size=0.2, random_state=42)
item_similarity = cosine_similarity(train_data.T.values)

#Model
def get_movie_recommendations(movie_name, num_recommendations=5):
    movie_index = train_data.columns.get_loc(movie_name)
    similarity_scores = item_similarity[movie_index]
    ranked_movies = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)
    recommendations = []
    for movie_index, score in ranked_movies:
        if train_data.columns[movie_index] != movie_name:
            recommendations.append(train_data.columns[movie_index])
            if len(recommendations) == num_recommendations:
                break
    return recommendations

#input
movie_name =input("Enter Movie Name: ") 
recommendations = get_movie_recommendations(movie_name)
print(f"Recommended movies for '{movie_name}':")
for movie in recommendations:
    print(movie)