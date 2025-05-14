import streamlit as st

def run():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    import urllib.parse
    import plotly.express as px
    import time

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3'
        }

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            if 'charset' in r.headers.get('content-type', '').lower():
                r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            body = soup.body
            if not body:
                return ""
            for tag in body(['script', 'style', 'meta', 'nav', 'footer', 'header']):
                tag.decompose()
            main_content = body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'article', 'section', 'div.content'])
            text = ' '.join([elem.get_text(strip=True) for elem in main_content]) if main_content else body.get_text(strip=True)
            text = re.sub(r'\s+', ' ', text)
            return text.strip() if len(text.strip()) > 100 else ""
        except Exception as e:
            st.error(f"Erreur lors de l'extraction de {url}: {str(e)}")
            return ""

    st.title("🔍 Semantic Analyzer")
    st.markdown("""
    Comparez les contenus de plusieurs pages web afin d'en extraire les mots-clés dominants,
    d'analyser leur similarité et de mettre en évidence des opportunités de contenu.
    """)

    urls_input = st.text_area("Entrez les URLs à comparer (une par ligne):", height=150)
    language = st.selectbox("Langue du contenu", ["french", "english"])

    if st.button("Analyser les contenus"):
        if urls_input:
            urls = []
            for line in urls_input.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith(('http://', 'https://')):
                    line = "https://" + line
                urls.append(line)

            if len(urls) < 2:
                st.error("Fournis au moins deux URLs valides.")
                return

            progress_text = st.empty()
            progress_bar = st.progress(0)

            texts = []
            valid_urls = []

            for i, url in enumerate(urls):
                progress_text.text(f"Extraction du contenu de {url}")
                progress_bar.progress((i+1)/len(urls))
                content = extract_content_from_url(url)
                if content:
                    texts.append(content)
                    valid_urls.append(url)
                else:
                    st.warning(f"L'URL {url} n'a pas fourni de contenu valide ou suffisant.")
                time.sleep(0.5)

            progress_bar.progress(1.0)
            progress_text.text("Extraction terminée")

            if len(valid_urls) < 2:
                st.error("Pas assez de contenu valide pour l'analyse. Au moins 2 URLs avec du contenu sont nécessaires.")
            else:
                st.success(f"Contenu extrait avec succès de {len(valid_urls)} URLs sur {len(urls)}.")
                cleaned_texts = [clean_text(text, language) for text in texts]

                try:
                    stop_words = stopwords.words(language)

                    # Expressions clés communes (2 à 4 mots)
                    st.subheader("🧩 Expressions clés communes (2 à 4 mots)")
                    ngram_vectorizer = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=1000)
                    ngram_matrix = ngram_vectorizer.fit_transform(cleaned_texts)
                    sum_words = ngram_matrix.sum(axis=0)
                    ngram_freq = [(word, int(sum_words[0, idx])) for word, idx in ngram_vectorizer.vocabulary_.items()]
                    ngram_freq_sorted = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
                    avg_per_doc = [(ng, freq, round(freq / len(valid_urls), 2)) for ng, freq in ngram_freq_sorted[:30]]
                    df_ngrams = pd.DataFrame(avg_per_doc, columns=["Expression", "Occurrences totales", "Occurrences moyennes"])
                    st.dataframe(df_ngrams, use_container_width=True)

                    # Analyse TF-IDF
                    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=500)
                    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    similarities = cosine_similarity(tfidf_matrix)

                    st.subheader("🔗 Similarité des contenus (Cosine)")
                    sim_df = pd.DataFrame(similarities, 
                                          index=[url.split('//')[1][:20] + '...' for url in valid_urls], 
                                          columns=[url.split('//')[1][:20] + '...' for url in valid_urls])
                    fig = px.imshow(sim_df,
                                   labels=dict(x="URL", y="URL", color="Similarité"),
                                   x=sim_df.columns,
                                   y=sim_df.index,
                                   color_continuous_scale="Blues")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📌 Mots-clés les plus importants")
                    cols = st.columns(min(3, len(valid_urls)))
                    for i, (url, vec) in enumerate(zip(valid_urls, tfidf_matrix.toarray())):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            st.markdown(f"**{url.split('//')[1][:30]}...**")
                            scores = dict(zip(feature_names, vec))
                            sorted_kws = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                            if sorted_kws:
                                kw_df = pd.DataFrame(sorted_kws, columns=["Mot-clé", "Score TF-IDF"])
                                st.dataframe(kw_df, use_container_width=True)
                            else:
                                st.write("Aucun mot-clé significatif trouvé.")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse TF-IDF: {str(e)}")
