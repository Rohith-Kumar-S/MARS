import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.recommender_utils import MovieRecommender 
import time
import os

if "user_data" not in st.session_state:
    st.session_state.user_data = {}
    
if "request_recommendations" not in st.session_state:
    st.session_state.request_recommendations = False


# Custom CSS for carousel styling
st.markdown("""
<style>
    .carousel-container {
        position: relative;
        padding: 20px 0;
    }
    .arrow-button {
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border: none;
        font-size: 24px;
        padding: 10px 15px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .arrow-button:hover {
        background-color: rgba(0, 0, 0, 0.8);
    }
    .movie-card {
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    .carousel-indicators {
        text-align: center;
        padding: 10px;
    }
    .indicator-dot {
        height: 10px;
        width: 10px;
        margin: 0 5px;
        background-color: #bbb;
        border-radius: 50%;
        display: inline-block;
        transition: background-color 0.3s ease;
    }
    .indicator-dot.active {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# Fetch movie data from API
def fetch_movie(title_id):
    """Fetch movie data from IMDB API with caching"""
    if title_id in st.session_state.movie_cache:
        # print('Fetching from cache')
        return st.session_state.movie_cache[title_id]
    
    try:
        response = requests.get(f"https://api.imdbapi.dev/titles/{title_id}", timeout=10)
        if response.status_code == 200:
            movie_data = response.json()
            st.session_state.movie_cache[title_id] = movie_data
            return movie_data
        else:
            # print(f"Error fetching movie {title_id} else: {response.status_code}")
            return None
    except Exception as e:
        # print(f"Error fetching movie {title_id} exp: {str(e)}")
        st.error(f"Error fetching movie {title_id}: {str(e)}")
        return None

# Create text for embedding
def create_embedding_text(movie_data):
    """Create combined text from title, plot, and genres for embedding"""
    title = movie_data.get('primaryTitle', '')
    plot = movie_data.get('plot', '')
    genres = ' '.join(movie_data.get('genres', []))
    
    # Combine with weights - plot is most important
    embedding_text = f"{title} {plot} {plot} {genres}"
    return embedding_text

# Get movie embedding
def get_movie_embedding(movie_data, model):
    """Get or create embedding for a movie"""
    movie_id = movie_data.get('id')
    
    if movie_id in st.session_state.embeddings_cache:
        return st.session_state.embeddings_cache[movie_id]
    
    text = create_embedding_text(movie_data)
    embedding = model.encode([text])[0]
    st.session_state.embeddings_cache[movie_id] = embedding
    return embedding

def fetch_similar_movies(fav_movie):
    fav_movie = fav_movie.iloc[0]
    res = requests.post('http://localhost:11434/api/embeddings',
                        json={
                            'model':'llama2',
                            'prompt': fav_movie['text_representation']
                        }
    )
    embedding = np.array([res.json()['embedding']], dtype='float32')
    D, I = st.session_state.vector_store.search(embedding, 5)
    # print(f"Similarity scores: {similarity}")
    # print(f"Indices: {I.flatten()}")
    return np.array(st.session_state.movies_df_cache['imdbId'])[I.flatten()]

def fetch_movie_id(imdb_id):
    """Fetch movie ID from IMDB ID"""
    try:
        return str(st.session_state.movies_df_cache[st.session_state.movies_df_cache['imdbId'] == imdb_id].iloc[0]['movieId'].astype(int))
    except IndexError:
        return None

# Display movie card
def display_movie_card(movie_data, col):
    """Display a movie card with image and title - entire card is clickable"""
    with col:
        # Create a clickable container with all movie info
        with st.container():
            # Create an invisible button that spans the entire card
            card_clicked = st.button(
                label="card",
                key=f"card_{movie_data['id']}",
                use_container_width=True,
                type="secondary",
                help=f"Click to rate {movie_data['primaryTitle']}"
            )
            
            if card_clicked:
                st.session_state.selected_movie = movie_data
                st.session_state.show_rating_dialog = True
                st.rerun()
            
            # Wrap everything in a div for styling
            with st.container():
                st.markdown('<div class="clickable-card">', unsafe_allow_html=True)
                
                # Get image URL or use placeholder
                image_url = None
                if movie_data.get('primaryImage'):
                    image_url = movie_data['primaryImage'].get('url')
                
                if not image_url:
                    image_url = "https://via.placeholder.com/300x450?text=No+Image"
                
                # Display the movie poster
                try:
                    st.image(image_url, use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                
                # Display title and year
                st.markdown(f"**{movie_data.get('primaryTitle', 'Unknown')}**")
                year = movie_data.get('startYear', 'N/A')
                st.caption(f"📅 {year}")
                
                # Display genres
                if movie_data.get('genres'):
                    genres_str = ", ".join(movie_data['genres'][:2])  # Show first 2 genres
                    st.caption(f"🎭 {genres_str}")
                
                # Display rating if exists
                if movie_data['id'] in st.session_state.ratings:
                    rating = st.session_state.ratings[movie_data['id']]
                    st.markdown(f"⭐ **Your rating: {rating}/5**")
                
                # Display IMDB rating
                if movie_data.get('rating'):
                    imdb_rating = movie_data['rating'].get('aggregateRating', 'N/A')
                    vote_count = movie_data['rating'].get('voteCount', 0)
                    st.caption(f"📊 IMDB: {imdb_rating}/10")
                    if vote_count > 1000:
                        st.caption(f"👥 {vote_count//1000}K votes")
                    else:
                        st.caption(f"👥 {vote_count} votes")
                
                st.markdown('</div>', unsafe_allow_html=True)

def update_click(current_index_key, val):
    st.session_state[current_index_key] = val

# Create carousel
def create_carousel(movies, carousel_key, movies_per_view=5):
    """Create a carousel with navigation arrows"""
    if not movies:
        st.info("No movies to display")
        return
    
    # Calculate total pages
    total_pages = (len(movies) - 1) // movies_per_view + 1
    
    # Get current index from session state
    current_index_key = f"{carousel_key}_index"
    if current_index_key not in st.session_state:
        st.session_state[current_index_key] = 0
    
    current_page = st.session_state[current_index_key]
    
    # Create navigation container
    nav_col1, nav_col2, nav_col3 = st.columns([1, 10, 1])
    
    # Left arrow
    with nav_col1:
        st.button("◀", key=f"{carousel_key}_left", 
                    disabled=(current_page == 0),
                    on_click=lambda a=current_index_key, b=max(0, current_page - 1): update_click(a,b),
                    help="Previous movies")
           
    # Right arrow
    with nav_col3:
        st.button("▶", key=f"{carousel_key}_right", 
                    disabled=(current_page >= total_pages - 1),
                    on_click=lambda a=current_index_key, b=min(total_pages - 1, current_page + 1): update_click(a,b),
                    help="Next movies")
    
    # Page indicator
    with nav_col2:
        st.markdown(f"<div style='text-align: center; padding: 10px;'>Page {current_page + 1} of {total_pages}</div>", 
                   unsafe_allow_html=True)
    
    # Display movies for current page
    start_idx = current_page * movies_per_view
    end_idx = min(start_idx + movies_per_view, len(movies))
    
    # Create columns for movies
    movie_cols = st.columns(movies_per_view)
    
    for i, col in enumerate(movie_cols):
        movie_idx = start_idx + i
        if movie_idx < len(movies):
            display_movie_card(movies[movie_idx], col)
        else:
            # Empty column for consistent spacing
            with col:
                st.empty()
    
    # Carousel indicators (dots)
    if total_pages > 1:
        indicator_html = '<div class="carousel-indicators">'
        for i in range(total_pages):
            active_class = "active" if i == current_page else ""
            indicator_html += f'<span class="indicator-dot {active_class}"></span>'
        indicator_html += '</div>'
        st.markdown(indicator_html, unsafe_allow_html=True)

def update_user_data(imdbId):
    """Update user data for the given IMDb ID"""
    if imdbId not in st.session_state.user_data:
        st.session_state.user_data[imdbId] = {}
    st.session_state.user_data[imdbId]['userId'] = st.session_state.user_id
    st.session_state.user_data[imdbId]['imdbId'] = imdbId
    st.session_state.user_data[imdbId]['rating'] = st.session_state.rating_slider
    st.session_state.user_data[imdbId]['movieId'] = str(int(fetch_movie_id(imdbId)))
        

# Rating dialog
def show_rating_dialog():
    """Show rating dialog for selected movie"""
    if st.session_state.show_rating_dialog and st.session_state.selected_movie:
        movie = st.session_state.selected_movie
        id = movie['id']
        print(id)
        similar_movies = fetch_similar_movies(st.session_state.movies_df_cache.loc[st.session_state.movies_df_cache['imdbId']==id])
        print(similar_movies)
        
        @st.dialog(f"Rate: {movie['primaryTitle']}")
        def rating_dialog():
            
            # Display movie image and details in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image_url = None
                if movie.get('primaryImage'):
                    image_url = movie['primaryImage'].get('url')
                if not image_url:
                    image_url = "https://via.placeholder.com/300x450?text=No+Image"
                try:
                    st.image(image_url, use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
            
            with col2:
                st.write(f"**{movie['primaryTitle']}**")
                
                if movie.get('startYear'):
                    st.write(f"📅 Year: {movie['startYear']}")
                
                if movie.get('genres'):
                    st.write(f"🎭 Genres: {', '.join(movie['genres'])}")
                
                if movie.get('plot'):
                    st.write("**Plot:**")
                    st.write(movie['plot'])
            
            st.divider()
            
            # Rating slider with visual stars
            st.write("**Your Rating:**")
            rating = st.slider("", 1, 5, 0, label_visibility="collapsed", key="rating_slider")
            st.markdown(f"### {'⭐' * rating}{'☆' * (5 - rating)}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Rating", type="primary", use_container_width=True, on_click=lambda imdbId=id: update_user_data(imdbId)):
                    st.session_state.ratings[movie['id']] = rating
                    st.session_state.show_rating_dialog = False
                    st.session_state.request_recommendations = True
                    st.success(f"Rated {rating}/5 ⭐")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_rating_dialog = False
                    st.session_state.rating_slider = 0
                    
            if similar_movies is not None:
                # Create navigation for similar movies carousel
                if 'dialog_carousel_index' not in st.session_state:
                    st.session_state.dialog_carousel_index = 0
                
                # Calculate pages for 3 movies at a time in dialog
                movies_per_page = 3
                total_pages = (len(similar_movies) - 1) // movies_per_page + 1
                current_page = st.session_state.dialog_carousel_index
                
                # Navigation buttons
                nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                
                with nav_col1:
                    if st.button("◀", key="dialog_left", disabled=(current_page == 0)):
                        st.session_state.dialog_carousel_index = max(0, current_page - 1)
                
                with nav_col3:
                    if st.button("▶", key="dialog_right", disabled=(current_page >= total_pages - 1)):
                        st.session_state.dialog_carousel_index = min(total_pages - 1, current_page + 1)
                
                with nav_col2:
                    st.markdown(f"<p style='text-align: center;'>Page {current_page + 1} of {total_pages}</p>", 
                               unsafe_allow_html=True)
                
                # Display current page of movies
                start_idx = current_page * movies_per_page
                end_idx = min(start_idx + movies_per_page, len(similar_movies))
                
                cols = st.columns(movies_per_page)
                for i, col in enumerate(cols):
                    movie_idx = start_idx + i
                    if movie_idx < len(similar_movies):
                        print(f"Fetching for: {similar_movies[movie_idx]}")
                        similar_movie = fetch_movie(similar_movies[movie_idx])
                        with col:
                            # Get image
                            sim_image_url = None
                            if similar_movie.get('primaryImage'):
                                sim_image_url = similar_movie['primaryImage'].get('url')
                            if not sim_image_url:
                                sim_image_url = "https://via.placeholder.com/200x300?text=No+Image"
                            
                            # Make movie clickable to switch to that movie
                            if st.button(
                                f"View",
                                key=f"similar_{similar_movie['id']}",
                                use_container_width=True,
                                help=f"View details for {similar_movie['primaryTitle']}"
                            ):
                                # Switch to the new movie
                                st.session_state.selected_movie = similar_movie
                                st.session_state.dialog_carousel_index = 0  # Reset carousel
                                st.rerun()
                            
                            # Display movie info
                            try:
                                st.image(sim_image_url, use_container_width=True)
                            except:
                                st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)
                            
                            st.caption(f"**{similar_movie.get('primaryTitle', 'Unknown')}**")
                            st.caption(f"📅 {similar_movie.get('startYear', 'N/A')}")
                            
                            # Show IMDB rating if available
                            if similar_movie.get('rating'):
                                imdb = similar_movie['rating'].get('aggregateRating', 'N/A')
                                st.caption(f"⭐ {imdb}/10")
            else:
                st.info("No similar movies found")
 
        
        rating_dialog()
        
        
def get_popular_movies(n=100):
    return st.session_state.ratings_df_cache.iloc[:n]

def create_representation(row_idx):
    # for row_idx in range(len(df)):
    return f"""
Title : {row_idx["title"]},
Genres : {", ".join(row_idx['genres'].split('|'))}"""

@st.cache_resource       
def load_data():
    movies = pd.read_csv(os.path.join(os.getcwd(), 'movies.csv'))
    links = pd.read_csv(os.path.join(os.getcwd(), 'links.csv'))
    st.session_state.ratings_df_cache =  pd.read_csv(os.path.join(os.getcwd(), 'top_100.csv'))
    df = pd.merge(movies, links, on='movieId', how='left')
    df['text_representation'] = df.apply(create_representation, axis=1)
    st.session_state.movies_df_cache = df
    st.session_state.data_loaded = True

# Main app
def main():
    if not st.session_state.data_loaded:
        load_data()
        print('loaded')
    # Sample movie IDs (you can expand this list)
    if st.session_state.popular_movies is None:
        st.session_state.popular_movies = get_popular_movies(n=20)
    sample_movie_ids = list(st.session_state.popular_movies['imdbId'].astype(str))
    print(sample_movie_ids)

    if st.session_state.request_recommendations and len(st.session_state.user_data.keys()) > 0:
        start_time = time.time()
        recommender = MovieRecommender(st.session_state.ratings_df_cache, st.session_state.user_data, st.session_state.vector_store, st.session_state.movies_df_cache)
        st.session_state.recommended_ids = recommender.get_hybrid_recommendations(st.session_state.user_id, alpha=0.7, n=20)
        print('recommendations generated: ', st.session_state.recommended_ids)
        st.session_state.request_recommendations = False
        print(f"Time taken for recommendations: {time.time() - start_time:.2f} seconds")

    # Sidebar for search and recommendations
    with st.sidebar:
        st.write(st.session_state.user_data)
        # st.write(st.session_state.rating_slider)
        # st.header("🔍 Search & Filter")
        
        # # Movie search by ID
        # search_id = st.text_input("Enter IMDB Title ID", placeholder="e.g., tt0114709")
        
        # if st.button("Search Movie", type="primary", use_container_width=True):
        #     if search_id:
        #         with st.spinner("Fetching movie..."):
        #             movie = fetch_movie(search_id)
        #             if movie:
        #                 st.success(f"Found: {movie['primaryTitle']}")
        #                 # Add to sample list if not present
        #                 if search_id not in sample_movie_ids:
        #                     sample_movie_ids.insert(0, search_id)
        #             else:
        #                 st.error("Movie not found!")
        
        # st.divider()
        
        # # Show recommendations based on selection
        # st.header("🎯 Get Recommendations")
        
        # available_movies = []
        # for movie_id in sample_movie_ids[:15]:  # Limit initial fetch
        #     movie = fetch_movie(movie_id)
        #     if movie:
        #         available_movies.append((movie_id, movie.get('primaryTitle', 'Unknown')))
        
        # if available_movies:
        #     selected_base = st.selectbox(
        #         "Select a movie:",
        #         options=[m[0] for m in available_movies],
        #         format_func=lambda x: next(m[1] for m in available_movies if m[0] == x)
        #     )
            
        #     if st.button("Find Similar Movies", type="primary", use_container_width=True):
        #         pass
        
        # st.divider()
        
        # # Display ratings summary
        # if st.session_state.ratings:
        #     st.header("⭐ Your Ratings")
        #     for movie_id, rating in list(st.session_state.ratings.items())[-5:]:  # Show last 5 ratings
        #         movie = fetch_movie(movie_id)
        #         if movie:
        #             st.write(f"**{movie['primaryTitle']}**")
        #             st.write(f"{'⭐' * rating}{'☆' * (5 - rating)}")
            
        #     if len(st.session_state.ratings) > 5:
        #         st.caption(f"...and {len(st.session_state.ratings) - 5} more")
    
    # Main content area
    if 'recommended_ids' in st.session_state and st.session_state.recommended_ids:
        st.header("🎯 Recommended Movies For You")
        st.markdown("Based on your selection, here are similar movies you might enjoy:")
        
        # Load recommended movies
        recommended_movies = []
        with st.spinner("Loading recommendations..."):
            for movie_id in st.session_state.recommended_ids:
                movie = fetch_movie(movie_id)
                if movie:
                    recommended_movies.append(movie)
        
        # Display in carousel
        create_carousel(recommended_movies, "rec_carousel", movies_per_view=5)
        
        # Add separator
        st.divider()
    
    # Display popular movies
    st.header("🔥 Popular Movies")
    st.markdown("Browse through our collection of popular movies:")
    
    # Load initial movies
    initial_movies = []
    with st.spinner("Loading movies..."):
        for imdb_id in sample_movie_ids:
            movie = fetch_movie(imdb_id)
            if movie:
                initial_movies.append(movie)
    
    # Display in carousel
    create_carousel(initial_movies, "main_carousel", movies_per_view=5)
    
    # Show rating dialog if triggered
    show_rating_dialog()
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Movie Recommender App | Data from IMDB API | Built with Streamlit</p>
            <p>Use the arrows to navigate through movies and click on any movie to rate it!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# if __name__ == "__main__":
main()