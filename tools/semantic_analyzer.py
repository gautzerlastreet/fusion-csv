import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests import Session
from bs4 import BeautifulSoup
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rake_nltk import Rake
import textstat
from typing import List, Tuple, Dict

# --- Configuration du logger structuré ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Session HTTP persistante ---
session: Session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; SemanticAnalyzer/1.0)'})

# --- Regex pré-compilés ---
PUNCT_PATTERN = re.compile(r"[^\w\s]")
DIGIT_PATTERN = re.compile(r"\d+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# --- Ressources NLTK ---
@st.cache_resource
def download_nltk_resources() -> None:
    """Télécharge punkt et stopwords si manquants."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# --- Liste d'expressions à exclure ---
EXCLUDED_EXPRESSIONS = {
    "bonjour", "merci", "au revoir", "salut", "bienvenue", "félicitations", "bravo",
    "cookies", "données personnelles", "caractère personnel", "protection des données", "mentions légales",
    "charte d’utilisation", "politique de confidentialité", "gérer les cookies", "stockées ou extraites",
    "gestion des cookies", "consentement aux cookies", "continuer sans accepter", "savoir plus", "en savoir plus",
    "utilisateur", "utilisateurs", "site web", "formulaire de contact", "the menu items", "avis", "blog", "guide d’achat",
    "newsletter", "rien à voir", "afin de", "valider votre inscription", "accéder au contenu",
    "page d’accueil", "prénom ou pseudo", "google llc", "envoyer des publicités", "adresse ip", "site", "email",
    "er", "css", "script", "footer", "header", "service client", "service spécifique"
}

def clean_text(text: str) -> str:
    """Nettoie le texte : minuscules, suppression ponctuation, chiffres, espaces."""
    txt = text.lower()
    txt = PUNCT_PATTERN.sub(' ', txt)
    txt = DIGIT_PATTERN.sub('', txt)
    txt = MULTI_SPACE_PATTERN.sub(' ', txt).strip()
    return txt


def is_relevant_expression(expr: str) -> bool:
    """Vérifie que l'expression contient au moins 2 mots et aucune exclusion."""
    if not isinstance(expr, str):
        return False
    expr_l = expr.lower()
    if len(expr_l.split()) < 2:
        return False
    return all(excl not in expr_l for excl in EXCLUDED_EXPRESSIONS)

@st.cache_data(ttl=3600)
def extract_content_from_url(url: str) -> Tuple[str, str, str, List[str], List[str]]:
    """
    Récupère depuis URL :
      - raw text des <p>,
      - <title>,
      - premier <h1>,
      - liste des <h2>,
      - liste des <h3>.
    """
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        body = soup.body or soup

        for tag in body(['script', 'style', 'nav', 'footer', 'header', 'form', 'table']):
            tag.decompose()

        raw_paras, h2s, h3s = [], [], []
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        h1 = ""
        for tag in body.find_all(['h1', 'h2', 'h3', 'p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not h1:
                h1 = txt
            elif tag.name == 'h2':
                h2s.append(txt)
            elif tag.name == 'h3':
                h3s.append(txt)
            else:
                raw_paras.append(txt)
        raw = ' '.join(raw_paras)
        cleaned = clean_text(raw)
        return raw, title, h1, h2s, h3s
    except Exception as e:
        logger.error(f"Erreur de récupération {url}: {e}")
        return "", "", "", [], []


def extract_rake_keywords(text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
    """
    Extrait les mots-clés RAKE et leurs scores.
    On utilise un découpage de phrases simple via regex pour éviter l’erreur punkt_tab.
    """
    rake = Rake()
    rake._tokenize_text_to_sentences = lambda txt: re.split(r'[.?!]\s+', txt)
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases_with_scores()[:max_keywords]


def get_readability_scores(text: str) -> Dict[str, float]:
    """Calcule Flesch, Kincaid et Gunning Fog."""
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text)
    }


def run() -> None:
    st.title('🔍 Semantic Analyzer')
    st.markdown(
        '''
        Comparez plusieurs pages pour extraire des expressions dominantes,
        repérer des opportunités de contenu et évaluer la pertinence SEO.
        '''
    )

    urls_input = st.text_area('Entrez les URLs (une par ligne)', height=150)
    language = st.selectbox('Langue du contenu', ['french', 'english'])

    if st.button('Analyser les contenus'):
        urls = [
            u.strip() if u.startswith(('http://', 'https://')) else f'https://{u.strip()}'
            for u in urls_input.splitlines() if u.strip()
        ]
        if len(urls) < 2:
            st.error('Veuillez fournir au moins 2 URLs.')
            return

        # 1) Scraping parallèle
        progress = st.progress(0)
        results: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(extract_content_from_url, url): url for url in urls}
            for i, fut in enumerate(as_completed(futures)):
                url = futures[fut]
                raw, title, h1, h2s, h3s = fut.result()
                if raw and len(raw) > 100:
                    results[url] = {'raw': raw, 'title': title, 'h1': h1, 'h2s': h2s, 'h3s': h3s}
                else:
                    st.warning(f"Contenu insuffisant : {url}")
                progress.progress((i + 1) / len(urls))

        if len(results) < 2:
            st.error('Pas assez de contenus validés.')
            return

        # 2) Statistiques de base + moyenne/médiane
        df_stats = pd.DataFrame([
            {'URL': u, 'Word Count': len(clean_text(data['raw']).split())}
            for u, data in results.items()
        ])
        median_wc = int(df_stats['Word Count'].median())
        mean_wc = int(df_stats['Word Count'].mean())
        st.info(f'💡 Nombre de mots médian attendu : {median_wc} | Nombre de mots moyen attendu : {mean_wc}')
        st.subheader('📊 Nombre de mots par URL')
        st.dataframe(df_stats, use_container_width=True)

        # Structure des pages
        df_struct = pd.DataFrame([
            {
                'URL': u,
                'Title': data['title'],
                'H1': data['h1'],
                'H2': '; '.join(data['h2s']),
                'H3': '; '.join(data['h3s'])
            }
            for u, data in results.items()
        ])
        st.subheader('🗂️ Structure des pages')
        st.dataframe(df_struct, use_container_width=True)

        docs = [clean_text(data['raw']) for data in results.values()]
        word_counts = df_stats['Word Count'].values
        stop_words = stopwords.words(language)

        # 3) CountVectorizer
        st.markdown(
            '''
            **Expressions clés (CountVectorizer)**  
            *Expression* : phrase extraite.  
            *Mean Count* : occurrence moyenne par document.  
            *Doc Coverage* : pourcentage de documents contenant l'expression.  
            *Density* : densité moyenne dans le contenu (en pourcentage).  
            '''
        )
        cv = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words)
        X = cv.fit_transform(docs)
        terms = cv.get_feature_names_out()
        coverage = np.array((X > 0).sum(axis=0)).ravel() / len(docs) * 100
        mask = coverage >= 40  # seuil 40%

        data_cv = []
        for term in terms[mask]:
            if is_relevant_expression(term):
                counts = X[:, cv.vocabulary_[term]].toarray().flatten()
                data_cv.append({
                    'Expression': term,
                    'Mean Count': int(round(np.mean(counts))),
                    'Doc Coverage (%)': int(round(coverage[cv.vocabulary_[term]])),
                    'Density (%)': int(round(np.mean(counts / word_counts * 100)))
                })
        df_cv = pd.DataFrame(data_cv).sort_values(['Doc Coverage (%)', 'Density (%)'], ascending=False)
        st.table(df_cv)

        # 4) TF-IDF
        st.markdown(
            '''
            **Top 20 TF-IDF**  
            *Expression* : phrase extraite.  
            *Avg TF-IDF* : score TF-IDF moyen sur les docs (sans unité).  
            '''
        )
        tfidf = TfidfVectorizer(ngram_range=(2, 4), stop_words=stop_words)
        Xtf = tfidf.fit_transform(docs)
        tf_terms = tfidf.get_feature_names_out()
        tf_scores = np.asarray(Xtf.mean(axis=0)).flatten()
        top_idx = np.argsort(tf_scores)[::-1][:20]
        df_tfidf = pd.DataFrame({
            'Expression': tf_terms[top_idx],
            'Avg TF-IDF': [round(score, 2) for score in tf_scores[top_idx]]
        })
        st.table(df_tfidf)

        # 5) RAKE Keywords
        st.markdown(
            '''
            **RAKE Keywords**  
            RAKE (Rapid Automatic Keyword Extraction) identifie des phrases clés significatives.  
            *Keyword* : phrase extraite.  
            *Score* : poids RAKE (score entier, plus élevé = plus pertinent).  
            '''
        )
        rake_rows = []
        for u, data in results.items():
            for phrase, score in extract_rake_keywords(data['raw']):
                rake_rows.append({
                    'URL': u,
                    'Keyword': phrase,
                    'Score': int(round(score))
                })
        df_rake = pd.DataFrame(rake_rows)
        st.table(df_rake)

        # 6) Readability Metrics
        st.markdown(
            '''
            **Readability Metrics**  
            *Flesch Reading Ease* : plus élevé = plus facile à lire.  
            *Flesch-Kincaid Grade* : niveau scolaire approximatif.  
            *Gunning Fog* : indice de complexité (plus élevé = plus complexe).  
            '''
        )
        read_rows = []
        for u, data in results.items():
            scores = get_readability_scores(data['raw'])
            read_rows.append({
                'URL': u,
                'Flesch Ease': int(round(scores['flesch_reading_ease'])),
                'Kincaid Grade': int(round(scores['flesch_kincaid_grade'])),
                'Gunning Fog': int(round(scores['gunning_fog']))
            })
        df_read = pd.DataFrame(read_rows)
        st.table(df_read)

        # 7) Export CSV
        st.download_button('Télécharger stats CSV', df_stats.to_csv(index=False), file_name='stats.csv')
        st.download_button('Télécharger CV CSV', df_cv.to_csv(index=False), file_name='countvectorizer.csv')
        st.download_button('Télécharger TF-IDF CSV', df_tfidf.to_csv(index=False), file_name='tfidf.csv')
        st.download_button('Télécharger RAKE CSV', df_rake.to_csv(index=False), file_name='rake.csv')
        st.download_button('Télécharger Readability CSV', df_read.to_csv(index=False), file_name='readability.csv')
