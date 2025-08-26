import pandas as pd
import requests
import numpy as np




class MovieRecommender:

    def __init__(self, top_100_data, embeddings,  user_data, vector_store, movies_df, label_encoder, onnx_session):
        self.top_100_data = top_100_data
        self.embeddings = embeddings
        self.user_data = user_data
        self.vector_store = vector_store
        self.movies_df = movies_df
        self.label_encoder = label_encoder
        self.onnx_session = onnx_session

        self.min_movies_with_ratings = 20
        self.rating_model = None

    def get_popular_movies(self, n=100, top_100=None):
        return self.top_100_data.iloc[:n]['imdbId'].values

    def process_user_data(self):
        return pd.DataFrame({
            'userId':   [v['userId'] for v in self.user_data.values()],
            'rating':  [float(v['rating']) for v in self.user_data.values()],
            'movieId':  [v['movieId'] for v in self.user_data.values()],
            'imdbId':   [v['imdbId'] for v in self.user_data.values()]
        })
        
    def get_movie_id_from_imdbId(self, imdbId):
        """Fetch movie data for a given IMDb ID"""
        movie = self.movies_df[self.movies_df['imdbId'] == imdbId]
        if not movie.empty:
            return movie.iloc[0]['movieId']
        return None


    def get_user_rated_movies(self, min_rating=4.0):
        user_data = self.process_user_data()
        sf = user_data[user_data['rating'] >= min_rating].sort_values(by='rating', ascending = False)[['imdbId', 'rating']]
        return list(sf.to_numpy())

    def fetch_similar_movies(self, imdbId, n=5):
        movie_index = self.movies_df[self.movies_df['imdbId'] == imdbId].index[0]
        embedding = self.embeddings[movie_index]
        results = self.vector_store.query(
            vector=embedding.tolist(),
            top_k=n
        )
        match_ids = []
        match_similarities = []
        for match in results['matches']:
            if match['id'] != imdbId:  # Skip the movie itself
                match_ids.append(match['id'])
                match_similarities.append(match['score'])
        return list(zip(match_ids, match_similarities))
        
    def movies_watched(self, user_data):
        movies_watched = set()
        for v in user_data.values():
            movies_watched.add(v['imdbId'])
        return movies_watched

    def get_user_recommendations_content_based(self):
        # Get movies the user has rated highly (e.g., >= 4 stars)
        user_movies = self.get_user_rated_movies(min_rating=3.8)
        movies_watched = self.movies_watched(self.user_data)
        # For each movie they liked, find similar movies
        similar_movies = {}
        similar = []
        for movie_id, rating in user_movies:
            # Your existing content-based function
            similar.extend(self.fetch_similar_movies(movie_id, n=20//len(user_movies)))
        # Weight by user's rating for that movie
        for sim_movie, similarity_score in similar:
            if sim_movie in movies_watched:
                continue
            weighted_score = similarity_score * (rating / 5.0)
            similar_movies[sim_movie] = max(similar_movies.get(sim_movie, -1), weighted_score)

        return sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
    

    def onxx_predict(self, imdbId):
        movieId = self.get_movie_id_from_imdbId(imdbId)
        movie_id = self.label_encoder.transform(np.array([movieId]))[0]
        user_id = np.array([-1], dtype=np.int64)   # must be numpy, not torch
        movie_id = np.array([movie_id], dtype=np.int64)
        inputs = {
            "user_ids": user_id,
            "movie_ids": movie_id
        }
        return  self.onnx_session.run(["predictions"], inputs)[0]


    def get_hybrid_recommendations(self, user_id, alpha=0.3, n=10):
        # 1. Get candidate movies from content-based
        cb_recommendations = self.get_user_recommendations_content_based()
        
        cb_movie_ids = [movie_id for movie_id, _ in cb_recommendations[:20]]
        # 2. Get CF predictions for those candidates + popular movies
        candidate_pool = cb_movie_ids 
        candidate_pool = list(set(candidate_pool))  # Remove duplicates
        
        # 3. Score all candidates with both methods
        hybrid_scores = []
        for movie_id in candidate_pool:
            cf_score = None
            try:
                cf_score = self.onxx_predict(movie_id) / 5.0
            except Exception as e:
                print(f"Error predicting CF score for {movie_id}: {e}")
            cb_score = dict(cb_recommendations).get(movie_id, None)
            if cf_score and cb_score:
                final_score = alpha * cf_score + (1-alpha) * cb_score
            elif cf_score:
                final_score = cf_score
            elif cb_score:
                final_score = cb_score
            else:
                final_score = None

            if final_score:
                hybrid_scores.append((movie_id, final_score))
        rec_movies = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:n]
        return [m[0] for m in rec_movies]
