import streamlit as st
import faiss
import os
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
    st.session_state.vector_store = faiss.read_index("basic_index.faiss")
st.set_page_config(page_title="Movie Recommender", layout="wide")
pages = {
    "": [
        st.Page(os.path.join("pages", "1_recommend.py"), title="Recommend"),
        st.Page(os.path.join("pages", "2_login.py"), title="Login")
    ]
}

pg = st.navigation(pages)
pg.run()