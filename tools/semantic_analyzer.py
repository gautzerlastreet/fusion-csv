import streamlit as st


def run():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
    import urllib.parse
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
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3'
        }
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            body = soup.body
            if not body:
                return ""
            for tag in body(['script', 'style', 'meta', 'nav', 'footer', 'header']):
                tag.decompose()
            main_content = body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'article', 'section', 'div.content'])
            text = ' '.join([elem.get_text(strip=True) for elem in main_content]) if main_content else body.get_text(strip=True)
            return re.sub(r'\s+', ' ', text.strip()) if len(text.strip()) > 100 else ""
        except Exception as e:
            st.error(f"Erreur lors de l'extraction de {url}: {str(e)}")
            return ""

    EXCLUDED_EXPRESSIONS = set([
        "er", "bonjour", "merci", "cookies", "données personnelles", "caractère personnel",
        "protection des données", "mentions légales", "charte d’utilisation", "politique de confidentialité",
        "gérer les cookies", "gestion des cookies", "stockées ou extraites", "consentement aux cookies",
        "retirer son consentement", "finalités statistiques anonymes", "finalités marketing similaires",
        "page d’accueil", "accéder au contenu", "aller au contenu", "menu", "top of page", "bottom of page",
        "skip to content", "passer au contenu", "tab to navigate", "to navigate through", "the menu items",
        "site web", "plan du site", "liens utiles", "voir la suite", "continuer sans accepter", "savoir plus",
        "lire la suite", "email", "mot de passe", "utilisateur", "utilisateurs", "site", "réseaux sociaux",
        "navigateur", "adresse IP", "google llc", "envoyer des publicités", "stocker des préférences",
        "service client", "service spécifique explicitement", "spécifique explicitement demandé", "rien à voir",
        "afin d’envoyer", "afin d’éviter", "raison pour laquelle", "mis à jour", "excellent chien", "prendre soin",
        "blog", "newsletter", "avis", "guide d’achat", "nouvelles fonctionnalités", "partagez votre expérience",
        "offre du moment", "laisser un commentaire", "prénom ou pseudo", "prendre rdv", "rendez-vous en ligne",
        "formulaire de contact", "accès à internet", "créer des profils", "annuler la réponse",
        "enregistrer les préférences", "auvergne", "cournon-d’auvergne", "rue andré theuriet",
        "rue des cordeliers", "rue morel ladeuil", "point da cruz",
        "consultez d’abord votre médecin", "ce site utilise des cookies", "en savoir plus sur nos services",
        "les champs obligatoires sont indiqués", "accéder à votre espace personnel",
        "offrir les meilleures expériences", "afin de vous garantir", "nous utilisons des cookies pour",
        "valider votre inscription", "consultez nos mentions légales", "nous utilisons des technologies",
        "ce site a été édité par", "prendre connaissance de notre politique", "facebook" 
    ])

    def is_relevant_expression(expr):
        if expr in EXCLUDED_EXPRESSIONS:
            return False
        if any(stop in expr for stop in EXCLUDED_EXPRESSIONS):
            return False
        if len(expr.split()) == 1:
            return False
        return True

    st.title("🔍 Semantic Analyzer")
    st.markdown("""
    Comparez les contenus de plusieurs pages web afin d'en extraire les mots-clés dominants,
    d'analyser leur similarité et de mettre en évidence des opportunités de contenu.
    """)

    urls_input = st.text_area("Entrez les URLs à comparer (une par ligne):", height=150)
    language = st.selectbox("Langue du contenu", ["french", "english"])

    if st.button("Analyser les contenus"):
        urls = ["https://" + u.strip() if not u.startswith("http") else u.strip() for u in urls_input.splitlines() if u.strip()]
        if len(urls) < 2:
            st.error("Fournis au moins deux URLs valides.")
            return

        progress_text = st.empty()
        progress_bar = st.progress(0)

        texts = []
        valid_urls = []
        word_counts = {}

        for i, url in enumerate(urls):
            progress_text.text(f"Extraction du contenu de {url}")
            progress_bar.progress((i+1)/len(urls))
            content = extract_content_from_url(url)
            if content:
                texts.append(content)
                valid_urls.append(url)
                word_counts[url] = len(content.split())
            else:
                st.warning(f"L'URL {url} n'a pas fourni de contenu valide ou suffisant.")
            time.sleep(0.5)

        progress_bar.progress(1.0)
        progress_text.text("Extraction terminée")

        if len(valid_urls) < 2:
            st.error("Pas assez de contenu valide pour l'analyse. Au moins 2 URLs avec du contenu sont nécessaires.")
            return

        st.success(f"Contenu extrait avec succès de {len(valid_urls)} URLs sur {len(urls)}.")
        cleaned_texts = [clean_text(text, language) for text in texts]

        stop_words = stopwords.words(language)

        st.subheader("📊 Nombre de mots par contenu")
        wc_df = pd.DataFrame(list(word_counts.items()), columns=["URL", "Nombre de mots"])
        st.dataframe(wc_df, use_container_width=True)

        st.subheader("🧩 Expressions clés communes (2 à 4 mots)")
        ngram_vectorizer = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=1000)
        ngram_matrix = ngram_vectorizer.fit_transform(cleaned_texts)
        sum_words = ngram_matrix.sum(axis=0)
        ngram_freq = [(word, int(sum_words[0, idx])) for word, idx in ngram_vectorizer.vocabulary_.items()]
        ngram_freq_filtered = [(word, freq) for word, freq in ngram_freq if is_relevant_expression(word)]
        ngram_freq_sorted = sorted(ngram_freq_filtered, key=lambda x: x[1], reverse=True)
        avg_per_doc = [(ng, freq, round(freq / len(valid_urls), 2)) for ng, freq in ngram_freq_sorted[:30]]
        df_ngrams = pd.DataFrame(avg_per_doc, columns=["Expression", "Occurrences totales", "Occurrences moyennes"])
        st.dataframe(df_ngrams, use_container_width=True)

        st.subheader("📌 Expressions les plus importantes (2 à 4 mots)")
        ngram_vectorizer_tf = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=500)
        tfidf_matrix = ngram_vectorizer_tf.fit_transform(cleaned_texts)
        feature_names = ngram_vectorizer_tf.get_feature_names_out()
        cols = st.columns(min(3, len(valid_urls)))

        for i, (url, vec) in enumerate(zip(valid_urls, tfidf_matrix.toarray())):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.markdown(f"**{url.split('//')[1][:30]}...**")
                scores = dict(zip(feature_names, vec))
                sorted_kws = [(k, v) for k, v in scores.items() if is_relevant_expression(k)]
                sorted_kws = sorted(sorted_kws, key=lambda x: x[1], reverse=True)[:10]
                if sorted_kws:
                    kw_df = pd.DataFrame(sorted_kws, columns=["Expression", "Score"])
                    st.dataframe(kw_df, use_container_width=True)
                else:
                    st.write("Aucune expression significative trouvée.")
