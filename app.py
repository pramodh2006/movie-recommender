import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset with safe encoding
@st.cache_data
def load_data():
    df = pd.read_csv("sample_movies_metadata.csv", encoding='ISO-8859-1')
    return df

# Get recommendations
def get_recommendations(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommendation System")
st.write("Get movie recommendations based on genres and keywords.")

# Load and prep data
df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'].fillna(''))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# User input
movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie:", sorted(movie_list))

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie, df, cosine_sim)
    if recommendations:
        st.subheader("You might also like:")
        for movie in recommendations:
            st.markdown(f"- {movie}")
    else:
        st.warning("Sorry, no recommendations found.")
