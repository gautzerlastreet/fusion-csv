import streamlit as st

def run():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize
    import urllib.parse
    import plotly.express as px
    import plotly.graph_objects as go
    import base64
    from io import BytesIO, StringIO
    import datetime
    from fpdf import FPDF
    import json
    from bertopic import BERTopic
    import umap
    import hdbscan

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    st.title("🔍 Semantic Analyzer")
    st.markdown("""
    Comparez les contenus de plusieurs pages web afin d'en extraire les mots-clés dominants,
    d'analyser leur similarité et de mettre en évidence des opportunités de contenu.
    """)

    urls_input = st.text_area("Entrez les URLs à comparer (une par ligne):", height=150)
    language = st.selectbox("Langue du contenu", ["french", "english"])
    lancer = st.button("Analyser les contenus")

    if lancer and urls_input:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        st.info("Extraction des contenus...")

        def extract_content_from_url(url):
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')
                for t in soup(['script', 'style']):
                    t.decompose()
                return soup.get_text()
            except:
                return ""

        texts = [extract_content_from_url(url) for url in urls]

        if sum(len(t) > 100 for t in texts) < 2:
            st.error("Pas assez de contenu valides pour analyse")
            return

        st.success("Contenus extraits. Analyse TF-IDF en cours...")

        vectorizer = TfidfVectorizer(stop_words=stopwords.words(language), max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        similarities = cosine_similarity(tfidf_matrix)
        st.subheader("🔗 Similarité des contenus (Cosine)")
        fig, ax = plt.subplots()
        sns.heatmap(similarities, annot=True, xticklabels=urls, yticklabels=urls, cmap="Blues")
        st.pyplot(fig)

        st.subheader("📌 Mots-clés les plus importants")
        for i, (url, vec) in enumerate(zip(urls, tfidf_matrix.toarray())):
            st.markdown(f"**{url}**")
            scores = dict(zip(feature_names, vec))
            sorted_kws = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
            df = pd.DataFrame(sorted_kws, columns=["Mot-clé", "Score TF-IDF"])
            st.dataframe(df, use_container_width=True)

        st.success("Analyse terminée.")
