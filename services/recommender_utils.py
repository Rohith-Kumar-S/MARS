import pandas as pd
import requests
import numpy as np




class MovieRecommender:

    def __init__(self, top_100_data, user_data, vector_store, movies_df):
        self.top_100_data = top_100_data
        self.user_data = user_data
        self.vector_store = vector_store
        self.movies_df = movies_df

        self.min_movies_with_ratings = 20
        self.rating_model = None

    def get_popular_movies(self, n=100, top_100=None):
        return list(self.top_100_data.sort_values(by='rating', ascending=False)['imdbId'].iloc[:n])

    def process_user_data(self):
        return pd.DataFrame({
            'userId':   [v['userId'] for v in self.user_data.values()],
            'movieId':  [v['movieId'] for v in self.user_data.values()],
            'rating':  [float(v['rating']) for v in self.user_data.values()],
            'imdbId':   [v['imdbId'] for v in self.user_data.values()]
        })

    def get_user_rated_movies(self, min_rating=4.0):
        user_data = self.process_user_data()
        sf = user_data[user_data['rating'] >= min_rating].sort_values(by='rating', ascending = False)[['imdbId', 'rating']]
        return list(sf.to_numpy())

    def create_representation(self, data):
        # for row_idx in range(len(df)):
        return f"""
Title : {data["title"]},
Genres : {", ".join(data['genres'])}"""

    def fetch_similar_movies(self, imdbId, vector_store, movies_df_cache, n=5):
        matches = movies_df_cache[movies_df_cache['imdbId'] == imdbId]
        fav_movie = matches.iloc[0] if not matches.empty else None
        if fav_movie is not None:
            res = requests.post('http://localhost:11434/api/embeddings',
                                json={
                                    'model':'llama2',
                                    'prompt': fav_movie['text_representation']
                                }
            )
            embedding = np.array([res.json()['embedding']], dtype='float32')
            D, I = vector_store.search(embedding, n)
            similarity = (1/(1+D))
            # print(f"Similarity scores: {similarity}")
            # print(f"Indices: {I.flatten()}")
            return zip(np.array(movies_df_cache['imdbId'])[I.flatten()], similarity[0])
        else:
            return []
        
    def movies_watched(self, user_data):
        movies_watched = set()
        for v in user_data.values():
            movies_watched.add(v['imdbId'])
        return movies_watched

    def get_user_recommendations_content_based(self):
        # Get movies the user has rated highly (e.g., >= 4 stars)
        user_movies = self.get_user_rated_movies(min_rating=4.0)
        movies_watched = self.movies_watched(self.user_data)
        print('movies_watched:', movies_watched)
        # For each movie they liked, find similar movies
        similar_movies = {}
        similar = []
        for movie_id, rating in user_movies:
            # Your existing content-based function
            similar.extend(self.fetch_similar_movies(movie_id, self.vector_store, self.movies_df, n=5))
        # Weight by user's rating for that movie
        print('movieId: ', movie_id)
        print(similar)
        for sim_movie, similarity_score in similar:
            if sim_movie in movies_watched:
                continue
            weighted_score = similarity_score * (rating / 5.0)
            if not similar_movies.get(sim_movie, False):
                similar_movies[sim_movie] = []
            similar_movies[sim_movie].append(weighted_score)
        # Average or max scores for each movie
        final_scores = {
            movie_id: np.mean(scores)  # or max(scores)
            for movie_id, scores in similar_movies.items()
        }
        
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


    def get_hybrid_recommendations(self, user_id, alpha=0.7, n=10):
        print('Loading recommendations...')
        # 1. Get candidate movies from content-based
        cb_recommendations = self.get_user_recommendations_content_based()
        print('cb recom: ', cb_recommendations)
        cb_movie_ids = [movie_id for movie_id, _ in cb_recommendations[:50]]
        
        # 2. Get CF predictions for those candidates + popular movies
        candidate_pool = cb_movie_ids + self.get_popular_movies(20)
        candidate_pool = list(set(candidate_pool))  # Remove duplicates
        
        # 3. Score all candidates with both methods
        hybrid_scores = []
        for movie_id in candidate_pool:
            cf_score = None
            if len(self.get_user_rated_movies())>=self.min_movies_with_ratings:
                cf_score = self.rating_model.predict(user_id, movie_id) / 5.0

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