import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import logging
import concurrent.futures
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import textstat
from urllib.parse import urlparse
from typing import Dict, List

# --- Logger Setup ---
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Multiple User Agents ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Mozilla/5.0 (X11; Linux x86_64)',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2 like Mac OS X)',
    'Mozilla/5.0 (Android 9; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0'
]

PATTERNS = {
    'punct': re.compile(r"[^\w\s]"),
    'digits': re.compile(r"\d+"),
    'spaces': re.compile(r"\s+")
}

EXCLUDED_EXPRESSIONS = {
    "bonjour", "merci", "cookies", "données personnelles", "caractère personnel",
    "site", "email", "newsletter", "utilisateur", "mentions légales", "page d'accueil",
    "er", "script", "footer", "header"
}

@st.cache_resource
def download_nltk_resources():
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f"{'tokenizers' if res=='punkt' else 'corpora'}/{res}")
        except LookupError:
            nltk.download(res)

def clean_text(text: str) -> str:
    if not text: return ""
    txt = text.lower()
    txt = PATTERNS['punct'].sub(' ', txt)
    txt = PATTERNS['digits'].sub('', txt)
    return PATTERNS['spaces'].sub(' ', txt).strip()

def is_relevant_expression(expr: str) -> bool:
    return isinstance(expr, str) and len(expr.split()) >= 2 and all(excl not in expr for excl in EXCLUDED_EXPRESSIONS)

def fetch_with_agents(url: str) -> requests.Response:
    for ua in USER_AGENTS:
        try:
            headers = {'User-Agent': ua}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and len(resp.text) > 100:
                return resp
        except Exception:
            continue
    raise ValueError("All user agents failed for " + url)

@st.cache_data(ttl=3600)
def extract_content_from_url(url: str) -> Dict:
    data = {k: 0 if k in ['images', 'tables', 'buttons', 'internal', 'external'] else '' for k in
            ['raw', 'title', 'h1', 'subsecs', 'images', 'tables', 'buttons', 'internal', 'external']}
    data['subsecs'] = []

    try:
        resp = fetch_with_agents(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        base = urlparse(url).netloc

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'form']):
            tag.decompose()

        container = soup.find('main') or soup.find(id='content') or soup.body
        data['images'] = len(container.find_all('img'))
        data['tables'] = len(container.find_all('table'))
        data['buttons'] = len(container.find_all('button'))

        int_links, ext_links = set(), set()
        for a in container.find_all('a', href=True):
            href = a['href']
            netloc = urlparse(href).netloc
            (ext_links if netloc and netloc != base else int_links).add(href)
        data['internal'], data['external'] = len(int_links), len(ext_links)

        for tag in container.find_all(['h1', 'h2', 'h3', 'p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not data['h1']:
                data['h1'] = txt
            elif tag.name in ['h2', 'h3']:
                data['subsecs'].append(f"{tag.name.upper()}: {txt}")
            elif tag.name == 'p':
                data['raw'] += ' ' + txt

        data['title'] = soup.title.string.strip() if soup.title and soup.title.string else ''
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
    return data

def get_readability_scores(text: str) -> Dict[str, int]:
    if not text: return {'Flesch Ease': 0, 'Kincaid Grade': 0, 'Gunning Fog': 0}
    return {
        'Flesch Ease': round(textstat.flesch_reading_ease(text)),
        'Kincaid Grade': round(textstat.flesch_kincaid_grade(text)),
        'Gunning Fog': round(textstat.gunning_fog(text))
    }

def process_content_vectors(docs: List[str], lang: str) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame(columns=['Expression', 'Mean Count', 'Coverage (%)'])

    stopw = stopwords.words(lang)
    cv = CountVectorizer(ngram_range=(2, 4), stop_words=stopw)
    X = cv.fit_transform(docs)
    terms = cv.get_feature_names_out()
    cov = np.array((X > 0).sum(axis=0)).ravel() / len(docs) * 100

    def contains_number_token(expr: str) -> bool:
        return any(re.fullmatch(r'\d+', word) for word in expr.split())

    data = []
    for i, term in enumerate(terms):
        counts = X[:, cv.vocabulary_[term]].toarray().ravel()
        nz = counts[counts > 0]
        mean_count = round(nz.mean(), 1)
        coverage = round(cov[i], 1)
        if nz.size and coverage >= 30 and is_relevant_expression(term) and not contains_number_token(term):
            if mean_count > 1 or coverage >= 40:
                data.append({'Expression': term, 'Mean Count': mean_count, 'Coverage (%)': coverage})

    return pd.DataFrame(data).sort_values('Coverage (%)', ascending=False).head(100)


def run():
    st.title('🔍 Semantic Analyzer')
    st.markdown('**Comparez plusieurs pages et comparez-les à votre page**')
    download_nltk_resources()

    urls_input = st.text_area('Entrez les URLs à comparer (une par ligne)', height=150)
    user_url = st.text_input('Entrez votre URL pour comparaison (facultatif)')
    language = st.selectbox('Langue du contenu', ['french', 'english'])

    if not st.button('Analyser'): return

    urls = [u.strip() if u.strip().startswith(('http://', 'https://')) else f"https://{u.strip()}"
            for u in urls_input.splitlines() if u.strip()]
    if len(urls) < 2:
        st.error('Veuillez fournir au moins 2 URLs.')
        return

    progress = st.progress(0)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_content_from_url, url): url for url in urls}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results[futures[future]] = future.result()
            progress.progress((i + 1) / len(urls))

    user_data = None
    if user_url:
        if not user_url.startswith(('http://', 'https://')):
            user_url = 'https://' + user_url.strip()
        user_data = extract_content_from_url(user_url)

    df_stats = pd.DataFrame([{'URL': u, 'Word Count': len(clean_text(d['raw']).split())} for u, d in results.items()])
    df_struct = pd.DataFrame([{'URL': u, 'Title': d['title'], 'H1': d['h1'], 'Structure': '\n'.join(d['subsecs'])} for u, d in results.items()])
    df_media = pd.DataFrame([{**{'URL': u}, **{k: d[k] for k in ['images', 'tables', 'buttons', 'internal', 'external']}} for u, d in results.items()])
    df_read = pd.DataFrame([{'URL': u, **get_readability_scores(d['raw'])} for u, d in results.items()])
    df_cv = process_content_vectors([clean_text(d['raw']) for d in results.values()], language)

    with st.expander('🗂️ Structure & Stats', expanded=True):
        st.dataframe(pd.merge(df_stats, df_struct, on='URL'), use_container_width=True)

    with st.expander('📊 Media & Links'):
        st.dataframe(df_media, use_container_width=True)

    with st.expander('🧩 Expressions clés'):
        st.dataframe(df_cv, use_container_width=True)

    with st.expander('📖 Readability Metrics'):
        st.dataframe(df_read, use_container_width=True)

    if user_data:
        uwc = len(clean_text(user_data['raw']).split())
        median_wc = int(df_stats['Word Count'].median())
        mean_wc = int(df_stats['Word Count'].mean())
        st.subheader('🔍 Analyse comparative de votre page')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Mots (votre page)', uwc, delta=uwc - mean_wc)
        with col2:
            st.metric('Médiane groupe', median_wc)

        missing = [t for t in df_cv['Expression'].tolist() if t not in clean_text(user_data['raw'])]
        if missing:
            st.markdown('**Mots clés manquants :**')
            st.table(pd.DataFrame({'Mots clés manquants': missing}))
        else:
            st.markdown('✅ Aucun mot clé manquant pertinent.')

        for col in ['images', 'internal', 'external']:
            if col not in df_media.columns:
                df_media[col] = 0
        media_mean = df_media[['images', 'internal', 'external']].mean().round().astype(int)
        imgs, intern, extern = user_data['images'], user_data['internal'], user_data['external']

        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            try:
                st.metric('Images (vous)', imgs, delta=int(imgs - media_mean['images']))
            except:
                st.metric('Images (vous)', imgs)
        with colm2:
            try:
                st.metric('Liens internes (vous)', intern, delta=int(intern - media_mean['internal']))
            except:
                st.metric('Liens internes (vous)', intern)
        with colm3:
            try:
                st.metric('Liens externes (vous)', extern, delta=int(extern - media_mean['external']))
            except:
                st.metric('Liens externes (vous)', extern)

        ur = get_readability_scores(user_data['raw'])
        mean_read = df_read.mean(numeric_only=True).round().astype(int)

        colr1, colr2, colr3 = st.columns(3)
        with colr1:
            try:
                st.metric('Flesch Ease (vous)', ur['Flesch Ease'], delta=int(ur['Flesch Ease'] - mean_read['Flesch Ease']))
            except:
                st.metric('Flesch Ease (vous)', ur['Flesch Ease'])
        with colr2:
            try:
                st.metric('Kincaid Grade (vous)', ur['Kincaid Grade'], delta=int(ur['Kincaid Grade'] - mean_read['Kincaid Grade']))
            except:
                st.metric('Kincaid Grade (vous)', ur['Kincaid Grade'])
        with colr3:
            try:
                st.metric('Gunning Fog (vous)', ur['Gunning Fog'], delta=int(ur['Gunning Fog'] - mean_read['Gunning Fog']))
            except:
                st.metric('Gunning Fog (vous)', ur['Gunning Fog'])

    st.download_button('📥 Export CSV Structure & Stats', pd.merge(df_stats, df_struct, on='URL').to_csv(index=False), file_name='structure_stats.csv')
    st.download_button('📥 Export CSV Media & Links', df_media.to_csv(index=False), file_name='media_links.csv')
    st.download_button('📥 Export CSV Expressions clés', df_cv.to_csv(index=False), file_name='expressions.csv')
    st.download_button('📥 Export CSV Readability', df_read.to_csv(index=False), file_name='readability.csv')

if __name__ == '__main__':
    run()
