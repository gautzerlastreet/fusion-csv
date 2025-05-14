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
from sklearn.feature_extraction.text import CountVectorizer
import textstat
from urllib.parse import urlparse
from typing import List, Dict, Tuple

# --- Logger Configuration ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- HTTP Session ---
session = Session()
session.headers.update({'User-Agent': 'SemanticAnalyzer/1.0'})

# --- Pre-compiled Regex Patterns ---
PUNCT_PATTERN = re.compile(r"[^\w\s]")
DIGIT_PATTERN = re.compile(r"\d+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# --- Download NLTK Resources ---
@st.cache_resource
def download_nltk_resources() -> None:
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f"tokenizers/{res}" if res=='punkt' else f"corpora/{res}")
        except LookupError:
            nltk.download(res)

download_nltk_resources()

# --- Excluded Expressions ---
EXCLUDED_EXPRESSIONS = {
    "bonjour","merci","au revoir","salut","bienvenue","félicitations","bravo",
    "cookies","données personnelles","caractère personnel","protection des données","mentions légales",
    "charte d’utilisation","politique de confidentialité","gérer les cookies","stockées ou extraites",
    "gestion des cookies","consentement aux cookies","continuer sans accepter","savoir plus","en savoir plus",
    "utilisateur","utilisateurs","site web","formulaire de contact","the menu items","avis","blog","guide d’achat",
    "newsletter","rien à voir","afin de","valider votre inscription","accéder au contenu",
    "page d’accueil","prénom ou pseudo","google llc","envoyer des publicités","adresse ip","site","email",
    "er","css","script","footer","header","service client","service spécifique"
}

# --- Utility Functions ---
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, digits, extra spaces."""
    txt = text.lower()
    txt = PUNCT_PATTERN.sub(' ', txt)
    txt = DIGIT_PATTERN.sub('', txt)
    txt = MULTI_SPACE_PATTERN.sub(' ', txt).strip()
    return txt


def is_relevant_expression(expr: str) -> bool:
    """At least 2 words and not in excluded list."""
    if not isinstance(expr, str):
        return False
    words = expr.lower().split()
    if len(words) < 2:
        return False
    return all(excl not in expr.lower() for excl in EXCLUDED_EXPRESSIONS)


@st.cache_data(ttl=3600)
def extract_content_from_url(url: str) -> Dict:
    """
    Scrape URL and extract:
      - raw text (paragraphs)
      - title, h1, subsections (H2/H3)
      - media counts (images, tables, buttons)
      - link counts (internal/external)
    """
    data = {
        'raw': '', 'title': '', 'h1': '', 'subsecs': [],
        'images': 0, 'tables': 0, 'buttons': 0,
        'internal': 0, 'external': 0
    }
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        base = urlparse(url).netloc
        # Remove irrelevant tags
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'form']):
            tag.decompose()
        # Media counts
        data['images'] = len(soup.find_all('img'))
        data['tables'] = len(soup.find_all('table'))
        data['buttons'] = len(soup.find_all('button'))
        # Link counts
        for a in soup.find_all('a', href=True):
            href = a['href']
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc != base:
                data['external'] += 1
            else:
                data['internal'] += 1
        # Text and structure
        paras = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not data['h1']:
                data['h1'] = txt
            elif tag.name in ['h2', 'h3']:
                data['subsecs'].append(f"{tag.name.upper()}: {txt}")
            elif tag.name == 'p':
                paras.append(txt)
        data['raw'] = ' '.join(paras)
        data['title'] = soup.title.string.strip() if soup.title and soup.title.string else ''
        return data
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return data


def get_readability_scores(text: str) -> Dict[str, int]:
    """Compute and round readability metrics."""
    return {
        'Flesch Ease': int(round(textstat.flesch_reading_ease(text))),
        'Kincaid Grade': int(round(textstat.flesch_kincaid_grade(text))),
        'Gunning Fog': int(round(textstat.gunning_fog(text)))
    }

# --- Main Streamlit App ---
def run() -> None:
    st.title('🔍 Semantic Analyzer')
    st.markdown('**Comparez plusieurs pages pour extraire expressions clés, structure, médias et liens**')

    urls_input = st.text_area('Entrez les URLs (une par ligne)', height=150)
    language = st.selectbox('Langue du contenu', ['french', 'english'])

    if not st.button('Analyser'):  # single button
        return
    urls = [u.strip() if u.startswith(('http://', 'https://')) else f"https://{u.strip()}"
            for u in urls_input.splitlines() if u.strip()]
    if len(urls) < 2:
        st.error('Veuillez fournir au moins 2 URLs.')
        return

    # Parallel scraping
    progress = st.progress(0)
    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_content_from_url, u): u for u in urls}
        for i, fut in enumerate(as_completed(futures)):
            url = futures[fut]
            results[url] = fut.result()
            progress.progress((i + 1) / len(urls))

    # Combined Structure & Stats
    df_comb = pd.DataFrame([{
        'URL': u,
        'Word Count': len(clean_text(d['raw']).split()),
        'Title': d['title'],
        'H1': d['h1'],
        'Structure': '<br>'.join(d['subsecs'])
    } for u, d in results.items()])
    with st.expander('🗂️ Structure & Stats', expanded=True):
        sty_comb = df_comb.style \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('white-space', 'normal')]}  
            ]) \
            .set_properties(subset=['URL', 'Title', 'H1', 'Structure'], **{'text-align': 'left', 'width': '250px'}) \
            .set_properties(subset=['Word Count'], **{'text-align': 'center', 'width': '80px'})
        st.write('Structure du contenu (Hn) et nombre de mots par URL')
        st.dataframe(sty_comb, use_container_width=True)

    # Media & Links
    df_media = pd.DataFrame([{
        'URL': u,
        'Images': d['images'],
        'Tables': d['tables'],
        'Buttons': d['buttons'],
        'Internal Links': d['internal'],
        'External Links': d['external']
    } for u, d in results.items()])
    with st.expander('📊 Media & Links', expanded=False):
        sty_media = df_media.style \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('white-space', 'normal')]}  
            ]) \
            .set_properties(subset=['URL'], **{'text-align': 'left', 'width': '200px'}) \
            .set_properties(subset=['Images', 'Tables', 'Buttons', 'Internal Links', 'External Links'], **{'text-align': 'center', 'width': '80px'})
        st.write('Comptage des médias et liens')
        st.dataframe(sty_media, use_container_width=True)

    # Expressions Clés (CountVectorizer)
    docs = [clean_text(d['raw']) for d in results.values()]
    stop_words = stopwords.words(language)
    cv = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words)
    X = cv.fit_transform(docs)
    terms = cv.get_feature_names_out()
    cov = np.array((X > 0).sum(axis=0)).ravel() / len(docs) * 100
    mask = cov >= 40
    rows = []
    for term in terms[mask][:40]:
        counts = X[:, cv.vocabulary_[term]].toarray().ravel()
        nz = counts[counts > 0]
        mean_count = int(round(nz.mean())) if nz.size > 0 else 0
        if is_relevant_expression(term):
            rows.append({'Expression': term, 'Mean Count': mean_count, 'Doc Coverage (%)': int(round(cov[cv.vocabulary_[term]]))})
    df_cv = pd.DataFrame(rows).sort_values('Doc Coverage (%)', ascending=False)
    with st.expander('🧩 Expressions clés', expanded=False):
        sty_cv = df_cv.style \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('white-space', 'normal')]}  
            ]) \
            .set_properties(subset=['Expression'], **{'text-align': 'left', 'width': '200px'}) \
            .set_properties(subset=['Mean Count', 'Doc Coverage (%)'], **{'text-align': 'center', 'width': '80px'})
        st.write('Top 40 des expressions clés (présence >= 40% des pages)')
        st.dataframe(sty_cv, use_container_width=True)

    # Readability Metrics
    df_read = pd.DataFrame([{'URL': u, **get_readability_scores(d['raw'])} for u, d in results.items()])
    mean_vals = df_read.mean(numeric_only=True).round().astype(int)
    col1, col2, col3 = st.columns(3)
    col1.metric('Moyenne Flesch Ease', mean_vals['Flesch Ease'])
    col2.metric('Moyenne Kincaid Grade', mean_vals['Kincaid Grade'])
    col3.metric('Moyenne Gunning Fog', mean_vals['Gunning Fog'])
    with st.expander('📖 Readability Metrics', expanded=False):
        sty_read = df_read.style \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('white-space', 'normal')]}  
            ]) \
            .set_properties(subset=['URL'], **{'text-align': 'left', 'width': '200px'}) \
            .set_properties(subset=['Flesch Ease', 'Kincaid Grade', 'Gunning Fog'], **{'text-align': 'center', 'width': '80px'})
        st.write('Scores de lisibilité par page')
        st.dataframe(sty_read, use_container_width=True)

    # Export CSVs
    st.download_button('📥 Export Structure & Stats', df_comb.to_csv(index=False), file_name='structure_stats.csv')
    st.download_button('📥 Export Media & Links', df_media.to_csv(index=False), file_name='media_links.csv')
    st.download_button('📥 Export Expressions', df_cv.to_csv(index=False), file_name='expressions.csv')
    st.download_button('📥 Export Readability', df_read.to_csv(index=False), file_name='readability.csv')
