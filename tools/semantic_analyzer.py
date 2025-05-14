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
from typing import List, Dict

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

# --- Regex Patterns ---
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
    txt = text.lower()
    txt = PUNCT_PATTERN.sub(' ', txt)
    txt = DIGIT_PATTERN.sub('', txt)
    txt = MULTI_SPACE_PATTERN.sub(' ', txt).strip()
    return txt


def is_relevant_expression(expr: str) -> bool:
    if not isinstance(expr, str): return False
    words = expr.lower().split()
    if len(words) < 2: return False
    return all(excl not in expr.lower() for excl in EXCLUDED_EXPRESSIONS)

@st.cache_data(ttl=3600)
def extract_content_from_url(url: str) -> Dict:
    data = {'raw':'','title':'','h1':'','subsecs':[],
            'images':0,'tables':0,'buttons':0,
            'internal':0,'external':0}
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text,'html.parser')
        base = urlparse(url).netloc
        for tag in soup(['script','style','header','footer','nav','form']):
            tag.decompose()
        container = soup.find('main') or soup.find(id='content')
        if not container:
            divs = soup.find_all('div')
            container = max(divs, key=lambda d: len(d.get_text().split()), default=soup.body)
        data['images'] = len(container.find_all('img'))
        data['tables'] = len(container.find_all('table'))
        data['buttons'] = len(container.find_all('button'))
        internal_links, external_links = set(), set()
        for a in container.find_all('a', href=True):
            href = a['href']
            parsed = urlparse(href)
            if parsed.netloc and parsed.netloc != base:
                external_links.add(href)
            else:
                internal_links.add(href)
        data['internal'] = len(internal_links)
        data['external'] = len(external_links)
        paras = []
        for tag in container.find_all(['h1','h2','h3','p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not data['h1']:
                data['h1'] = txt
            elif tag.name in ['h2','h3']:
                data['subsecs'].append(f"{tag.name.upper()}: {txt}")
            elif tag.name == 'p':
                paras.append(txt)
        data['raw'] = ' '.join(paras)
        data['title'] = soup.title.string.strip() if soup.title and soup.title.string else ''
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
    return data


def get_readability_scores(text: str) -> Dict[str,int]:
    return {
        'Flesch Ease': int(round(textstat.flesch_reading_ease(text))),
        'Kincaid Grade': int(round(textstat.flesch_kincaid_grade(text))),
        'Gunning Fog': int(round(textstat.gunning_fog(text)))
    }

# --- Main Streamlit App ---

def run() -> None:
    st.title('🔍 Semantic Analyzer')
    st.markdown('**Comparez plusieurs pages et comparez-les à votre page**')

    urls_input = st.text_area('Entrez les URLs à comparer (une par ligne)', height=150)
    user_url = st.text_input("Entrez votre URL pour comparaison (facultatif)")
    language = st.selectbox('Langue du contenu', ['french','english'])

    if not st.button('Analyser'): return
    urls = [u.strip() if u.startswith(('http://','https://')) else f"https://{u.strip()}"
            for u in urls_input.splitlines() if u.strip()]
    if len(urls) < 2:
        st.error('Veuillez fournir au moins 2 URLs.')
        return

    # Scrape competitor pages
    progress = st.progress(0)
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_content_from_url, u): u for u in urls}
        for i, fut in enumerate(as_completed(futures)):
            url = futures[fut]
            results[url] = fut.result()
            progress.progress((i+1)/len(urls))

    # Scrape user page if given
    user_data = None
    if user_url:
        if not user_url.startswith(('http://','https://')):
            user_url = 'https://' + user_url.strip()
        user_data = extract_content_from_url(user_url)

    # Build DataFrames
    df_stats = pd.DataFrame([{'URL':u, 'Word Count': len(clean_text(d['raw']).split())}
                             for u, d in results.items()])
    df_media = pd.DataFrame([{'URL':u, 'Images': d['images'], 'Tables': d['tables'], 'Buttons': d['buttons'],
                               'Internal': d['internal'], 'External': d['external']}
                              for u, d in results.items()])
    docs = [clean_text(d['raw']) for d in results.values()]
    stopw = stopwords.words(language)
    cv = CountVectorizer(ngram_range=(2,4), stop_words=stopw)
    X = cv.fit_transform(docs)
    terms = cv.get_feature_names_out()
    cov = np.array((X>0).sum(axis=0)).ravel() / len(docs) * 100
    rows = []
    for term in terms[cov>=40][:40]:
        counts = X[:, cv.vocabulary_[term]].toarray().ravel()
        nz = counts[counts>0]
        mean_ct = int(round(nz.mean())) if nz.size>0 else 0
        if is_relevant_expression(term):
            rows.append({'Expression':term, 'Mean Count':mean_ct, 'Coverage (%)': int(round(cov[cv.vocabulary_[term]]))})
    df_cv = pd.DataFrame(rows).sort_values('Coverage (%)', ascending=False)
    df_read = pd.DataFrame([{'URL':u, **get_readability_scores(d['raw'])}
                             for u, d in results.items()])

    # Display Sections
    with st.expander('🗂️ Structure & Stats', expanded=True):
        df_str = pd.DataFrame([{'URL':u, 'Title':d['title'], 'H1':d['h1'], 'Structure': ' | '.join(d['subsecs'])}
                                for u, d in results.items()])
        st.dataframe(pd.merge(df_stats, df_str, on='URL'), use_container_width=True)
    with st.expander('📊 Media & Links', expanded=False):
        st.dataframe(df_media, use_container_width=True)
    with st.expander('🧩 Expressions clés', expanded=False):
        st.dataframe(df_cv, use_container_width=True)

    mean_vals = df_read.mean(numeric_only=True).round().astype(int)
    m1, m2, m3 = st.columns(3)
    m1.metric('Moy. Flesch Ease', mean_vals['Flesch Ease'])
    m2.metric('Moy. Kincaid Grade', mean_vals['Kincaid Grade'])
    m3.metric('Moy. Gunning Fog', mean_vals['Gunning Fog'])
    with st.expander('📖 Readability Metrics', expanded=False):
        st.dataframe(df_read, use_container_width=True)

    # Export CSVs
    st.download_button('📥 Export Structure & Stats',
                       pd.merge(df_stats, df_str, on='URL').to_csv(index=False),
                       file_name='structure_stats.csv')
    st.download_button('📥 Export Media & Links', df_media.to_csv(index=False), file_name='media_links.csv')
    st.download_button('📥 Export Expressions', df_cv.to_csv(index=False), file_name='expressions.csv')
    st.download_button('📥 Export Readability', df_read.to_csv(index=False), file_name='readability.csv')

    # Comparative Summary if user_url provided (moved after all analyses)
    if user_data:
        st.subheader('🔍 Analyse comparative de votre page')

        # Word Count Comparison
        uwc = len(clean_text(user_data['raw']).split())
        median_wc = int(df_stats['Word Count'].median())
        mean_wc = int(df_stats['Word Count'].mean())
        c1, c2 = st.columns(2)
        c1.metric('Mots (votre page)', uwc, delta=int(uwc - mean_wc))
        c2.metric('Médiane groupe', median_wc)

        # Key Expressions Missing (filtered)
        stopw_terms = df_cv[(df_cv['Mean Count'] > 1) & (df_cv['Coverage (%)'] > 50)]['Expression'].tolist()
        user_text = clean_text(user_data['raw'])
        missing = [t for t in stopw_terms if t not in user_text]
        if missing:
            st.table(pd.DataFrame({'Mots clés manquants': missing}))
        else:
            st.write('Aucun mot clé manquant pertinent.')

        # Media & Links Comparison
        media_mean = df_media[['Images', 'Internal', 'External']].mean().round().astype(int)
        u_imgs = user_data['images']
        u_int = user_data['internal']
        u_ext = user_data['external']
        m1, m2, m3 = st.columns(3)
        m1.metric('Images (vous)', u_imgs, delta=int(u_imgs - media_mean['Images']))
        m2.metric('Liens internes (vous)', u_int, delta=int(u_int - media_mean['Internal']))
        m3.metric('Liens externes (vous)', u_ext, delta=int(u_ext - media_mean['External']))

        # Readability Comparison
        ur = get_readability_scores(user_data['raw'])
        mean_read = df_read.mean(numeric_only=True).round().astype(int)
        r1, r2, r3 = st.columns(3)
        r1.metric('Flesch Ease (vous)', ur['Flesch Ease'], delta=int(ur['Flesch Ease'] - mean_read['Flesch Ease']))
        r2.metric('Kincaid Grade (vous)', ur['Kincaid Grade'], delta=int(ur['Kincaid Grade'] - mean_read['Kincaid Grade']))
        r3.metric('Gunning Fog (vous)', ur['Gunning Fog'], delta=int(ur['Gunning Fog'] - mean_read['Gunning Fog']))

if __name__ == "__main__":
    run()
