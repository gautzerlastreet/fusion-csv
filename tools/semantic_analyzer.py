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
    import numpy as np

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

    EXCLUDED_EXPRESSIONS = set([
        "bonjour", "merci", "au revoir", "salut", "bienvenue", "félicitations", "bravo",
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
            for tag in body(['script', 'style', 'meta', 'nav', 'footer', 'header', 'h4', 'h5', 'table']):
                tag.decompose()
            content = []
            skip = False
            for tag in body.find_all(True):
                if tag.name in ['h4', 'h5']:
                    skip = True
                elif tag.name.startswith('h'):
                    skip = False
                elif not skip and tag.name == 'p':
                    content.append(tag.get_text(" ", strip=True))
            return re.sub(r'\s+', ' ', " ".join(content))
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

        st.subheader("📊 Nombre de mots par URL")
        wc_df = pd.DataFrame(list(word_counts.items()), columns=["URL", "Nombre de mots"])
        st.dataframe(wc_df, use_container_width=True)

        st.subheader("🧩 Expressions clés communes (2 à 4 mots)")
        vec = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words, max_features=3000)
        X = vec.fit_transform(cleaned_texts)
        features = vec.get_feature_names_out()
        X_array = X.toarray()
        total_docs = len(valid_urls)

        data = []
        all_word_counts = list(word_counts.values())
        total_words = sum(all_word_counts)

        for i, expr in enumerate(features):
            counts = X_array[:, i]
            total_occurrence = counts.sum()
            doc_count = (counts > 0).sum()
            if is_relevant_expression(expr) and (doc_count / total_docs) >= 0.4:
                moyenne = round(total_occurrence / total_docs, 2)
                min_occur = int(counts.min())
                max_occur = int(counts.max())
                moyenne_fmt = f"{moyenne} ({max_occur}-{min_occur})"
                couverture = round((doc_count / total_docs) * 100)
                densite = round((total_occurrence / total_words) * 100, 2)
                data.append((expr, moyenne_fmt, couverture, densite))

        df_final = pd.DataFrame(data, columns=["Expression", "Moyenne par contenu", "% Présence", "Densité moyenne"])
        df_final = df_final[df_final["% Présence"] >= 40]
        df_final = df_final.sort_values(by=["% Présence", "Densité moyenne", "Moyenne par contenu"], ascending=[False, False, False]).reset_index(drop=True)

        df_final["% Présence"] = df_final["% Présence"].astype(str) + "%"
        df_final["Densité moyenne"] = df_final["Densité moyenne"].astype(str) + "%"

        st.dataframe(df_final, use_container_width=True)

        # Statistiques globales
        st.subheader("📈 Statistiques globales")
        med_words = int(np.median(all_word_counts))
        avg_words = int(np.mean(all_word_counts))

        st.markdown(f"**Nombre médian de mots :** {med_words}")
        st.markdown(f"**Nombre moyen de mots :** {avg_words}")

        st.markdown("**Top 10 des expressions les plus stratégiques :**")
        top_10 = df_final.head(20).reset_index(drop=True)
        st.dataframe(top_10, use_container_width=True)
