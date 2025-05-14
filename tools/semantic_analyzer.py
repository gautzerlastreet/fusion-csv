import streamlit as st

def run():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    from nltk.corpus import stopwords
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

    # Expressions à exclure
    EXCLUDED_EXPRESSIONS = set([
        # Interactions et messages inutiles
        "bonjour", "merci", "au revoir", "salut", "bienvenue", "félicitations", "bravo",
        # Mentions légales / cookies
        "cookies", "données personnelles", "caractère personnel", "protection des données", "mentions légales",
        "charte d’utilisation", "politique de confidentialité", "gérer les cookies", "stockées ou extraites",
        "gestion des cookies", "consentement aux cookies", "continuer sans accepter", "savoir plus", "en savoir plus",
        "utilisateur", "utilisateurs", "site web", "formulaire de contact", "the menu items", "avis", "blog", "guide d’achat",
        "newsletter", "rien à voir", "afin de", "valider votre inscription", "accéder au contenu",
        "page d’accueil", "prénom ou pseudo", "google llc", "envoyer des publicités", "adresse ip", "site", "email",
        "er", "css", "script", "footer", "header", "service client", "service spécifique"
    ])

    def is_relevant_expression(expr):
        if not isinstance(expr, str):
            return False
        expr = expr.lower()
        if len(expr.split()) < 2:
            return False
        for excl in EXCLUDED_EXPRESSIONS:
            if excl in expr:
                return False
        return True

    def deduplicate_ngrams(ngrams):
        """Supprime les expressions plus courtes incluses dans des plus longues"""
        deduped = []
        for expr, score in sorted(ngrams, key=lambda x: (-len(x[0].split()), -x[1])):
            if not any(expr in longer for longer, _ in deduped):
                deduped.append((expr, score))
        return deduped

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_content_from_url(url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.body
            if not body:
                return ""
            for tag in body(['script', 'style', 'meta', 'nav', 'footer', 'header']):
                tag.decompose()
            content = body.get_text(separator=" ", strip=True)
            return re.sub(r'\s+', ' ', content)
        except Exception as e:
            st.error(f"Erreur pour {url} : {str(e)}")
            return ""

    st.title("🔍 Semantic Analyzer")
    st.markdown("""
    Comparez les contenus de plusieurs pages web afin d'en extraire les expressions dominantes,
    repérer les opportunités de contenu, et analyser leur pertinence SEO.
    """)

    urls_input = st.text_area("Entrez les URLs à comparer (une par ligne):", height=150)
    language = st.selectbox("Langue du contenu", ["french", "english"])

    if st.button("Analyser les contenus"):
        urls = ["https://" + u.strip() if not u.startswith("http") else u.strip() for u in urls_input.splitlines() if u.strip()]
        if len(urls) < 2:
            st.error("Fournissez au moins deux URLs valides.")
            return

        progress_text = st.empty()
        progress_bar = st.progress(0)
        texts, valid_urls, word_counts = [], [], {}

        for i, url in enumerate(urls):
            progress_text.text(f"Extraction du contenu de {url}")
            progress_bar.progress((i + 1) / len(urls))
            content = extract_content_from_url(url)
            if content and len(content) > 100:
                texts.append(content)
                valid_urls.append(url)
                word_counts[url] = len(content.split())
            else:
                st.warning(f"Contenu insuffisant ou vide : {url}")
            time.sleep(0.5)

        progress_bar.progress(1.0)
        progress_text.text("Extraction terminée ✅")

        if len(valid_urls) < 2:
            st.error("Pas assez d'URLs avec contenu pour lancer l'analyse.")
            return

        cleaned_texts = [clean_text(txt) for txt in texts]
        stop_words = stopwords.words(language)

        # Nombre de mots
        st.subheader("📊 Nombre de mots par URL")
        wc_df = pd.DataFrame(list(word_counts.items()), columns=["URL", "Nombre de mots"])
        st.dataframe(wc_df, use_container_width=True)

        # Expressions communes
        st.subheader("🧩 Expressions clés communes (2 à 4 mots)")
        vec = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=1000)
        X = vec.fit_transform(cleaned_texts)
        sums = X.sum(axis=0)
        ngrams = [(word, int(sums[0, idx])) for word, idx in vec.vocabulary_.items()]
        filtered = [(ng, cnt) for ng, cnt in ngrams if is_relevant_expression(ng) and cnt > 1]
        deduped = deduplicate_ngrams(filtered)
        df_occ = pd.DataFrame([(ng, cnt, round(cnt/len(valid_urls), 2)) for ng, cnt in deduped],
                              columns=["Occurrence", "Présence", "Moyenne"])
        st.dataframe(df_occ, use_container_width=True)

        # Expressions par URL
        st.subheader("📌 Expressions les plus importantes (2 à 4 mots)")
        vec2 = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=500)
        matrix = vec2.fit_transform(cleaned_texts)
        features = vec2.get_feature_names_out()
        cols = st.columns(min(3, len(valid_urls)))
        for i, (url, vec) in enumerate(zip(valid_urls, matrix.toarray())):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.markdown(f"**{url.split('//')[1][:30]}...**")
                scores = dict(zip(features, vec))
                sorted_exprs = sorted([(k, v) for k, v in scores.items() if is_relevant_expression(k) and v > 0],
                                      key=lambda x: x[1], reverse=True)
                top_exprs = deduplicate_ngrams(sorted_exprs)[:10]
                if top_exprs:
                    df_top = pd.DataFrame(top_exprs, columns=["Expression", "Score"])
                    st.dataframe(df_top, use_container_width=True)
                else:
                    st.write("Aucune expression significative trouvée.")
