import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer    # type: ignore
from sklearn.metrics.pairwise import cosine_similarity         # type: ignore

# Sample dataset of movies
movies_data = {
    'MovieID': [1, 2, 3, 4, 5],
    'Title': ['Inception', 'Titanic', 'Avatar', 'The Matrix', 'The Godfather'],
    'Genres': ['Sci-Fi Action', 'Romance Drama', 'Sci-Fi Adventure', 'Sci-Fi Action', 'Crime Drama']
}

# User's liked movie
user_liked_movie = 'Inception'

# Convert to DataFrame
movies_df = pd.DataFrame(movies_data)

# Create a count vectorizer for genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies_df['Genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Create a DataFrame for similarity scores
similarity_df = pd.DataFrame(cosine_sim, index=movies_df['Title'], columns=movies_df['Title'])

def recommend_movies(user_movie, num_recommendations=3):
    
    # Get similarity scores for the user_liked_movie
    similar_movies = similarity_df[user_movie].sort_values(ascending=False)

    # Exclude the movie itself and return top recommendations
    recommendations = similar_movies.iloc[1:num_recommendations + 1].index.tolist()
    return recommendations

# Example usage
recommended_movies = recommend_movies(user_liked_movie)
print(f"Movies recommended for you if you liked '{user_liked_movie}': {recommended_movies}")
