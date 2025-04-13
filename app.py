
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

# ---------------------------
# 1. Load your Excel data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("clustered_session_summaries.xlsx")
    
    df = df[['SerialNo','Cleaned_Summary']].dropna().reset_index(drop=True)
    return df

df = load_data()

# ---------------------------
# 2. Load Sentence-BERT model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# 3. Encode summaries and cluster
# ---------------------------
@st.cache_data
def embed_and_cluster(df, k=19):
    summaries = df['Cleaned_Summary'].tolist()
    embeddings = model.encode(summaries, show_progress_bar=True)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return np.array(embeddings), cluster_labels, kmeans.cluster_centers_

embeddings, cluster_labels, cluster_centroids = embed_and_cluster(df)
df['Cluster'] = cluster_labels

# ---------------------------
# 4. Streamlit App Interface
# ---------------------------
st.title("📚 Session Summary Finder")

user_input = st.text_input("🔍 Enter topic-related keywords (comma-separated):", "")

if user_input:
    with st.spinner("Finding relevant session..."):
        # Embed the user's query
        query_embedding = model.encode(user_input, convert_to_tensor=True)

        # Find most relevant cluster (session)
        sims = [util.cos_sim(query_embedding, c).item() for c in cluster_centroids]
        best_cluster_id = int(np.argmax(sims))

        st.success(f"🔎 Most relevant session: Session {best_cluster_id}")

        # Filter summaries in that session
        session_df = df[df['Cluster'] == best_cluster_id].copy()

        # Rank summaries within session
        session_df['Similarity'] = session_df['Cleaned_Summary'].apply(
            lambda x: util.cos_sim(model.encode(x), query_embedding).item()
        )

        top3 = session_df.sort_values(by='Similarity', ascending=False).head(3)

        st.subheader("📘 Top 3 Relevant Summaries:")
        for i, row in top3.iterrows():
            st.text_area(f"Summary {i+1} (Serial {row['SerialNo']})", value=row['Cleaned_Summary'], height=150)
