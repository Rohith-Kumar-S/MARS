import streamlit as st
import requests
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from services.recommender_utils import MovieRecommender 
import time
import os
import pickle
import onnxruntime as ort
import random

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Initialize session state
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'show_rating_dialog' not in st.session_state:
    st.session_state.show_rating_dialog = False
if 'movie_cache' not in st.session_state:
    st.session_state.movie_cache = {}
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'carousel_index' not in st.session_state:
    st.session_state.carousel_index = 0
if 'rec_carousel_index' not in st.session_state:
    st.session_state.rec_carousel_index = 0
if 'movies_df_cache' not in st.session_state:
    st.session_state.movies_df_cache = None
if 'ratings_df_cache' not in st.session_state:
    st.session_state.ratings_df_cache = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'popular_movies' not in st.session_state:
    st.session_state.popular_movies = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'user_id' not in st.session_state:
        st.session_state.user_id = 30000
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
        return st.session_state.movie_cache[title_id]
    
    try:
        response = requests.get(f"https://api.imdbapi.dev/titles/{title_id}", timeout=10)
        if response.status_code == 200:
            movie_data = response.json()
            st.session_state.movie_cache[title_id] = movie_data
            return movie_data
        else:
            return None
    except Exception as e:
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

def fetch_similar_movies(movie_index):
    embedding = st.session_state.embeddings[movie_index]
    results = st.session_state.vector_store.query(
        vector=embedding.tolist(),
        top_k=10
    )
    match_ids = {match['id'] for match in results['matches']}
    return list(match_ids)

# def fetch_movie_id(imdb_id):
#     """Fetch movie ID from IMDB ID"""
#     try:
#         return str(st.session_state.movies_df_cache[st.session_state.movies_df_cache['imdbId'] == imdb_id].iloc[0]['movieId'].astype(int))
#     except IndexError:
#         return None

# Display movie card
def display_movie_card(movie_data, col, rec_carousel):
    """Display a movie card with image and title - entire card is clickable"""
    with col:
        # Create a clickable container with all movie info
        with st.container():
            # Create an invisible button that spans the entire card
            # Wrap everything in a div for styling
            with st.container():
                st.markdown('<div class="clickable-card">', unsafe_allow_html=True)
                
                # Get image URL or use placeholder
                image_url = None
                if movie_data.get('primaryImage'):
                    image_url = movie_data['primaryImage'].get('url')
                
                if not image_url:
                    image_url = "https://via.placeholder.com/300x450?text=No+Image"
                
                st.markdown(
                    """
                    <style>
                    .clickable-img {
                        height: 280px !important;
                        width: auto !important;
                        object-fit: cover;
                        border-radius: 10px;
                        cursor: pointer;
                        transition: 0.3s;
                    }
                    .clickable-img:hover {
                        opacity: 0.8;
                        transform: scale(1.02);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f'<a href="https://www.imdb.com/title/{movie_data.get("id", "")}" target="_blank"><img src="{image_url}" class="clickable-img"></a>', unsafe_allow_html=True)
                # try:
                #     st.image(image_url, use_container_width=True)
                # except:
                #     st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                
                # Display title and year
                # st.markdown(f"**{movie_data.get('primaryTitle', 'Unknown')}**")
                # st.markdown('</div>', unsafe_allow_html=True)
        card_clicked = st.button(
                label="Rate Movie",
                key=f"card_{movie_data['id']}" if rec_carousel == "main_carousel" else f"recom_{movie_data['id']}",
                use_container_width=True,
                type="secondary",
                help=f"Click to rate {movie_data['primaryTitle']}"
            )
            
        if card_clicked:
            st.session_state.selected_movie = movie_data
            st.session_state.show_rating_dialog = True
            st.rerun()
            
                
                

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
    nav_col1, nav_col2, nav_col3 = st.columns([1, 20, 1], vertical_alignment='center')
    
    # Left arrow
    with nav_col1:

        if st.button("◀", key=f"{carousel_key}_left", 
                    disabled=(current_page == 0),
                    on_click=lambda a=current_index_key, b=max(0, current_page - 1): update_click(a,b),
                    help="Previous movies"):
            st.session_state.show_rating_dialog = False
            st.rerun()
           
    # Right arrow
    with nav_col3:
        if st.button("▶", key=f"{carousel_key}_right", 
                    disabled=(current_page >= total_pages - 1),
                    on_click=lambda a=current_index_key, b=min(total_pages - 1, current_page + 1): update_click(a,b),
                    help="Next movies"):
            st.session_state.show_rating_dialog = False
            st.rerun()

    # # Page indicator
    # with nav_col2:
    #     st.markdown(f"<div style='text-align: center; padding: 10px;'>Page {current_page + 1} of {total_pages}</div>", 
    #                unsafe_allow_html=True)
    
    # Display movies for current page
    start_idx = current_page * movies_per_view
    end_idx = min(start_idx + movies_per_view, len(movies))
    with nav_col2.container():
        # Create columns for movies
        movie_cols = st.columns(movies_per_view)
        
        for i, col in enumerate(movie_cols):
            movie_idx = start_idx + i
            if movie_idx < len(movies):
                display_movie_card(movies[movie_idx], col, carousel_key)
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
    st.session_state.user_data[imdbId]['movieId'] = int(get_movie_id_from_imdbId(imdbId))
        
def get_movie_id_from_imdbId(imdbId):
    """Fetch movie data for a given IMDb ID"""
    movie = st.session_state.movies_df_cache[st.session_state.movies_df_cache['imdbId'] == imdbId]
    if not movie.empty:
        return movie.iloc[0]['movieId']
    return None

# Rating dialog
def show_rating_dialog():
    """Show rating dialog for selected movie"""
    if st.session_state.show_rating_dialog and st.session_state.selected_movie:
        movie = st.session_state.selected_movie
        id = movie['id'] 
        matches = st.session_state.movies_df_cache.loc[st.session_state.movies_df_cache['imdbId']==id]
        similar_movies = []
        if len(matches)>0:
            similar_movies = fetch_similar_movies(matches.index[0])
        
        @st.dialog(f"Rate {movie['primaryTitle']}")
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
            rating = st.slider("", 1.0, 5.0, 0.0, label_visibility="collapsed", key="rating_slider")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Rating", type="primary", use_container_width=True, on_click=lambda imdbId=id: update_user_data(imdbId)):
                    st.session_state.ratings[movie['id']] = rating
                    st.session_state.show_rating_dialog = False
                    st.session_state.request_recommendations = True
                    st.success(f"Rated {rating}/5 ⭐")
                    st.rerun()
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_rating_dialog = False
                    st.session_state.rating_slider = 0
                    st.rerun()
                    
            if similar_movies is not None:
                st.write("**Similar Movies**")
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
                        st.rerun()
                        
                
                with nav_col3:
                    if st.button("▶", key="dialog_right", disabled=(current_page >= total_pages - 1)):
                        st.session_state.dialog_carousel_index = min(total_pages - 1, current_page + 1)
                        st.rerun()
                
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
                                st.session_state.show_rating_dialog = True 
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
    movies = pd.read_csv(os.path.join(os.getcwd(), 'data','processed_movies.csv'))
    ratings_df = pd.read_csv(os.path.join(os.getcwd(), 'data','top_100.csv'))
    embeddings = np.load('data/embeddings.npz')['embeddings']
    with open("data/movie_label_encoder.pkl", "rb") as f:
        lbl_movie = pickle.load(f)
    onnx_session = ort.InferenceSession("data/mars_mov_quantized1.onnx")
    pc = Pinecone(api_key="pcsk_74k399_TT9sjm5THx4nEak9GGTqnZp5zicVcLj8n7K4sBJSeS3uxsucyYiLNLt7SFoTMbw")
    vector_store = pc.Index("phantom-recom-light")
    return movies, ratings_df, embeddings, lbl_movie, onnx_session, vector_store

# Main app
def main():
    with st.sidebar:
        st.title("Movie Recommender")
        
        search_id = st.text_input("Enter IMDB Title ID", placeholder="e.g., tt0114709")
        
        if st.button("Search Movie", type="primary", use_container_width=True):
            if search_id:
                with st.spinner("Fetching movie..."):
                    movie = fetch_movie(search_id)
                    if movie:
                        # Add to sample list if not present
                        st.session_state.show_rating_dialog = True
                        st.session_state.selected_movie = movie
                    else:
                        st.error("Movie not found!")
        
    if not st.session_state.data_loaded:
        movies, ratings_df, embeddings, lbl_movie, onnx_session, vector_store = load_data()

        st.session_state.movies_df_cache = movies
        st.session_state.ratings_df_cache = ratings_df
        st.session_state.embeddings = embeddings
        st.session_state.lbl_movie = lbl_movie
        st.session_state.onnx_session = onnx_session
        st.session_state.vector_store = vector_store
        st.session_state.data_loaded = True
    # Sample movie IDs (you can expand this list)
    if st.session_state.popular_movies is None:
        st.session_state.popular_movies = get_popular_movies(n=20)
    sample_movie_ids = list(st.session_state.popular_movies['imdbId'].astype(str))
    
    
    if st.session_state.request_recommendations and len(st.session_state.user_data.keys()) > 0:
        start_time = time.time()
        recommender = MovieRecommender(st.session_state.ratings_df_cache, st.session_state.embeddings, st.session_state.user_data, st.session_state.vector_store, st.session_state.movies_df_cache, st.session_state.lbl_movie, st.session_state.onnx_session)
        st.session_state.recommended_ids = recommender.get_hybrid_recommendations(st.session_state.user_id, alpha=0.3, n=40)
        st.session_state.request_recommendations = False


    # Main content area
    if 'recommended_ids' in st.session_state and st.session_state.recommended_ids:
        st.header("Best Picks For You")
        
        # Load recommended movies
        recommended_movies = []
        with st.spinner("Loading recommendations..."):
            for movie_id in st.session_state.recommended_ids:
                movie = fetch_movie(movie_id)
                if movie:
                    recommended_movies.append(movie)
        
        # Display in carousel
        create_carousel(recommended_movies, "rec_carousel", movies_per_view=5)
    
    
    # Display popular movies
    st.header("Top 100 Movies")
    
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

if __name__ == "__main__":
    main()