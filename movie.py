import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ====== PAGE CONFIGURATION ======
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")

# ====== PAGE HEADER ======
st.title("🎥 Movie Recommendation System")
st.markdown("##### A Machine Learning based Personalized Movie Suggestion Platform using Collaborative Filtering & SVD")
st.markdown("---")

# ====== LOAD DATA ======
@st.cache_data
def load_data():
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# ====== DATA PREPROCESSING ======
min_user_ratings = 5
min_movie_ratings = 5

user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()

valid_users = user_counts[user_counts >= min_user_ratings].index
valid_movies = movie_counts[movie_counts >= min_movie_ratings].index

ratings = ratings[ratings['userId'].isin(valid_users) & ratings['movieId'].isin(valid_movies)]

unique_user_ids = ratings['userId'].unique()
unique_movie_ids = ratings['movieId'].unique()

user2idx = {u: i for i, u in enumerate(unique_user_ids)}
idx2user = {i: u for u, i in user2idx.items()}
movie2idx = {m: i for i, m in enumerate(unique_movie_ids)}
idx2movie = {i: m for m, i in movie2idx.items()}

n_users = len(user2idx)
n_items = len(movie2idx)

R = np.zeros((n_users, n_items))
for row in ratings.itertuples():
    R[user2idx[row.userId], movie2idx[row.movieId]] = row.rating

movie_id_to_title = movies.set_index('movieId')['title'].to_dict()

# ====== SIMILARITY MATRICES ======
item_sim = cosine_similarity(R.T)
user_sim = cosine_similarity(R)

# ====== RECOMMENDATION FUNCTIONS ======
def recommend_itemCF(userId, n=10):
    if userId not in user2idx:
        return []
    u = user2idx[userId]
    user_row = R[u]
    scores = item_sim.dot(user_row)
    seen = set(ratings[ratings['userId'] == userId]['movieId'])
    movie_scores = [(scores[movie2idx[mid]], mid) for mid in unique_movie_ids if mid not in seen]
    movie_scores = sorted(movie_scores, reverse=True)
    return [mid for s, mid in movie_scores[:n]]

def recommend_userCF(userId, n=10):
    if userId not in user2idx:
        return []
    u = user2idx[userId]
    scores = user_sim[u].dot(R)
    seen = set(ratings[ratings['userId'] == userId]['movieId'])
    movie_scores = [(scores[movie2idx[mid]], mid) for mid in unique_movie_ids if mid not in seen]
    movie_scores = sorted(movie_scores, reverse=True)
    return [mid for s, mid in movie_scores[:n]]

def recommend_SVD(userId, n=10, k=50):
    if userId not in user2idx:
        return []
    svd = TruncatedSVD(n_components=k, random_state=42)
    movie_factors = svd.fit_transform(R.T)
    u = user2idx[userId]
    user_vec = R[u]
    profile = user_vec.dot(movie_factors)
    scores = movie_factors.dot(profile)
    seen = set(ratings[ratings['userId'] == userId]['movieId'])
    movie_scores = [(scores[movie2idx[mid]], mid) for mid in unique_movie_ids if mid not in seen]
    movie_scores = sorted(movie_scores, reverse=True)
    return [mid for s, mid in movie_scores[:n]]

# ====== SIDEBAR CONTROLS ======
st.sidebar.header("⚙️ Settings")
userId = st.sidebar.number_input("Enter User ID", min_value=int(ratings['userId'].min()), max_value=int(ratings['userId'].max()), step=1)
algo = st.sidebar.selectbox("Select Recommendation Algorithm", ["Item-Based CF", "User-Based CF", "SVD (k=50)"])
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Try different algorithms and compare how the recommendations change!")

# ====== GENERATE RECOMMENDATIONS ======
if st.sidebar.button("🎯 Get Recommendations"):
    with st.spinner("Generating personalized recommendations..."):
        if algo == "Item-Based CF":
            recs = recommend_itemCF(userId, top_n)
        elif algo == "User-Based CF":
            recs = recommend_userCF(userId, top_n)
        else:
            recs = recommend_SVD(userId, top_n, k=50)

    if len(recs) == 0:
        st.warning("No recommendations found. Try a different user ID.")
    else:
        st.success(f"✅ Top {top_n} Recommended Movies for User ID {userId} ({algo})")
        for i, mid in enumerate(recs, 1):
            st.markdown(f"**{i}. {movie_id_to_title[mid]}** 🎬")

st.markdown("---")
st.caption("Developed by A. Lokesh | Machine Learning Recommender Project | MovieLens Dataset © GroupLens")

