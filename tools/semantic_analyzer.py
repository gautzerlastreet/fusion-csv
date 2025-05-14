import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import time
import nltk
from nltk.corpus import stopwords

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

def clean_text(text, language):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_content_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/html',
        'Accept-Language': 'fr-FR'
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style', 'meta', 'nav', 'footer', 'header']):
            tag.decompose()
        text = ' '.join([elem.get_text(strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"Erreur pour {url}: {e}")
        return ""

def run():
    st.title("🔍 Semantic Analyzer")
    st.markdown("Compare plusieurs pages web pour identifier les mots-clés dominants et les opportunités de contenu.")

    urls_input = st.text_area("Entrez les URLs à comparer (une par ligne):", height=150)
    language = st.selectbox("Langue du contenu", ["french", "english"])

    if st.button("Analyser les contenus") and urls_input:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip().startswith(('http://', 'https://'))]
        if len(urls) < 2:
            st.error("Fournis au moins deux URLs valides.")
            return

        progress = st.progress(0)
        texts = []
        valid_urls = []

        for i, url in enumerate(urls):
            st.write(f"Extraction depuis : {url}")
            content = extract_content_from_url(url)
            if content and len(content) > 100:
                texts.append(clean_text(content, language))
                valid_urls.append(url)
            else:
                st.warning(f"{url} ignorée (contenu insuffisant)")
            time.sleep(0.3)
            progress.progress((i + 1) / len(urls))

        if len(texts) < 2:
            st.error("Pas assez de contenus valides.")
            return

        # Analyse TF-IDF
        stop_words = stopwords.words(language)
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=500)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Matrice de similarité
        st.subheader("🔗 Similarité des contenus (cosine)")
        labels = [u.split("//")[1][:30] + '...' for u in valid_urls]
        sim_df = pd.DataFrame(cosine_sim, index=labels, columns=labels)
        fig = px.imshow(sim_df, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Mots-clés importants
        st.subheader("📌 Mots-clés dominants")
        cols = st.columns(min(3, len(valid_urls)))
        for i, vec in enumerate(tfidf_matrix.toarray()):
            top_kws = sorted(zip(feature_names, vec), key=lambda x: x[1], reverse=True)[:10]
            df = pd.DataFrame(top_kws, columns=["Mot-clé", "Poids"])
            with cols[i % len(cols)]:
                st.markdown(f"**{labels[i]}**")
                st.dataframe(df, use_container_width=True)

        # Analyse thématique
        st.subheader("🧩 Analyse thématique")
        count_vectorizer = CountVectorizer(stop_words=stop_words, max_features=500)
        count_matrix = count_vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=min(3, len(valid_urls)), random_state=42)
        lda.fit(count_matrix)
        count_features = count_vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(lda.components_):
            top_words = [(count_features[i], topic[i]) for i in topic.argsort()[:-11:-1]]
            df = pd.DataFrame(top_words, columns=["Mot", "Poids"])
            st.markdown(f"**Topic {topic_idx+1}**")
            fig = px.bar(df, x="Mot", y="Poids")
            st.plotly_chart(fig, use_container_width=True)
