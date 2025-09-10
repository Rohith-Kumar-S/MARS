# Multi-domain Adaptive Recommendation System (MARS)

Hybrid recommendation system combining collaborative and content-based filtering, trained on Movie Lens dataset.

## Overview

This project implements a hybrid recommendation system that solves the cold start problem while providing accurate personalized recommendations for existing users. It combines:
- **Neural Collaborative Filtering** for learning user-movie interaction patterns
- **Content-Based Filtering** using semantic embeddings for movie similarity
- **Real-time movie information** from IMDb API
- **Optimized inference** using ONNX Runtime

## Features

- **Hybrid Recommendations**: Seamlessly blends collaborative and content-based filtering
- **Cold Start Handling**: New users get content-based recommendations immediately
- **Similar Movies**: Find movies similar to any selected movie using semantic search
- **Real-time Data**: Fetches latest movie information, posters, and ratings from IMDb
- **Fast Inference**: ONNX-optimized model for millisecond-level predictions
- **Scalable Vector Search**: Pinecone database handles 87,000+ movie vectors efficiently
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  Recommendation  │────▶│   IMDb API      │
│                 │     │     Engine       │     │   (Metadata)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌──────────────────┐
          │ Collaborative   │    │  Content-Based   │
          │   Filtering     │    │    Filtering     │
          │  (ONNX Model)   │    │   (Pinecone)     │
          └─────────────────┘    └──────────────────┘
                    │                      │
                    ▼                      ▼
          ┌─────────────────┐    ┌──────────────────┐
          │ Neural Network  │    │ Ollama2 Embeddings│
          │   (PyTorch)     │    │  (512 dims PCA)  │
          └─────────────────┘    └──────────────────┘
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch with CUDA acceleration
- **Embeddings**: Ollama2 for semantic movie representations
- **Vector Database**: Pinecone for similarity search
- **Model Optimization**: ONNX Runtime for inference
- **Dimensionality Reduction**: PCA (4096 → 512 dimensions)
- **API Integration**: IMDb API for movie metadata
- **Web Framework**: Streamlit for interactive UI
- **Data Source**: MovieLens dataset (22,000+ movies, 87,000+ ratings)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
## Usage

### Running the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## Performance

### Model Metrics
- **RMSE**: 0.82 on test set

### Storage Optimization
- **Original Embeddings**: 1.2 GB (4096 dimensions)
- **Optimized Embeddings**: 98 MB (512 dimensions via PCA)
- **Storage Reduction**: 91.8%
- **Similarity Search**: <100ms for 87,000 vectors

## Project Structure

```
MARS/
├── app.py                  # Streamlit application
├── model/
│   └── model.py
├── data/
│   ├── embeddings.npz      # Movie embeddings from Movie lens dataset
│   ├── mars_mov_quantized1.onnx  # Optimized Model
│   └── movie_label_encoder.pkl   # Data encoder
|   └── processed_movies.csv      # Combined dataset with movies and ratings
|   └── top_100.csv  # Top 100 movies
├── services/
│   ├── recommender_utils.py # Processes user data and provides recommendation
├── requirements.txt
└── README.md
```

## Technical Details

### Neural Collaborative Filtering Model
Architecture Overview:
```python
- Dual-path design: GMF + MLP
- Total parameters: ~1.1M 
- Input: User ID, Movie ID
- Output: Predicted rating (1-5 scale)
```

```python
# GMF Path (Generalized Matrix Factorization)
- User embeddings: 32 dimensions
- Movie embeddings: 32 dimensions
- Interaction: Element-wise multiplication
- Output: 32-dimensional interaction vector

# MLP Path (Multi-Layer Perceptron)
- User embeddings: 64 dimensions
- Movie embeddings: 64 dimensions
- Concatenation: 128 dimensions (64 + 64)
- Hidden layers:
  └─ Linear: 128 → 64 (ReLU activation)
  └─ Linear: 64 → 32 (ReLU activation)
- Output: 32-dimensional feature vector

# Final Fusion Layer
- Concatenation: GMF (32) + MLP (32) = 64 dimensions
- Output layer: Linear 64 → 1
- Rating prediction: Scaled to [0.5, 5] range
```

### Content-Based Filtering
```python
- Text representation: Movie title + genres + tags
- Embedding model: Ollama2 (LLaMA 2 based)
- Original dimensions: 4096
- Reduced dimensions: 512 (via PCA)
- Similarity metric: Cosine similarity
- Vector database: Pinecone (87,585 vectors)
```

### Hybrid Strategy
```python
if user_rating_count == 0:
    return popular_movies()
else:
    return hybrid_recommendations(
        cf_weight=0.3,
        cb_weight=0.7
    )
```

## Future Improvements
- Imporve model complexity
- Add support for music and book recommendations
- Recommendations based on user activity

## Acknowledgments

- MovieLens dataset provided by GroupLens Research
- IMBD Data for movies
