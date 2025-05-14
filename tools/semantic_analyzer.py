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
from typing import List, Tuple, Dict

# --- Configuration du logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Session HTTP persistante ---
session: Session = requests.Session()
session.headers.update({'User-Agent': 'SemanticAnalyzer/1.0'})

# --- Regex pré-compilés ---
PUNCT_PATTERN = re.compile(r"[^\w\s]")
DIGIT_PATTERN = re.compile(r"\d+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# --- Télécharger ressources NLTK ---
@st.cache_resource
def download_nltk_resources() -> None:
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f"tokenizers/{res}" if res=='punkt' else f"corpora/{res}")
        except LookupError:
            nltk.download(res)

download_nltk_resources()

# --- Liste des expressions exclues ---
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

# --- Fonctions utilitaires ---
def clean_text(text: str) -> str:
    txt = text.lower()
    txt = PUNCT_PATTERN.sub(' ', txt)
    txt = DIGIT_PATTERN.sub('', txt)
    txt = MULTI_SPACE_PATTERN.sub(' ', txt).strip()
    return txt


def is_relevant_expression(expr: str) -> bool:
    if not isinstance(expr, str):
        return False
    words = expr.lower().split()
    if len(words) < 2:
        return False
    return all(excl not in expr.lower() for excl in EXCLUDED_EXPRESSIONS)


def extract_content_from_url(url: str) -> Tuple[str,str,str,List[str]]:
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        body = soup.body or soup
        for tag in body(['script','style','nav','footer','header','form','table']):
            tag.decompose()
        raw_paras = []
        title = soup.title.string.strip() if soup.title and soup.title.string else ''
        h1 = ''
        subsections: List[str] = []
        for tag in body.find_all(['h1','h2','h3','p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not h1:
                h1 = txt
            elif tag.name in ['h2','h3']:
                subsections.append(f"{tag.name.upper()}: {txt}")
            elif tag.name == 'p':
                raw_paras.append(txt)
        raw = ' '.join(raw_paras)
        cleaned = clean_text(raw)
        return raw, title, h1, subsections
    except Exception as e:
        logger.error(f"Erreur récupération {url}: {e}")
        return '', '', '', []


def get_readability_scores(text: str) -> Dict[str,int]:
    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    fog = textstat.gunning_fog(text)
    return {'Flesch Ease':int(round(ease)), 'Kincaid Grade':int(round(grade)), 'Gunning Fog':int(round(fog))}


# --- Fonction principale Streamlit ---
def run() -> None:
    st.title('🔍 Semantic Analyzer')
    st.markdown('**Comparez plusieurs pages pour extraire expressions clés & analyser SEO**')

    urls_input = st.text_area('Entrez les URLs (une par ligne)', height=120)
    language = st.selectbox('Langue', ['french','english'])

    if not st.button('Analyser les contenus'):
        return

    urls = [u.strip() if u.startswith(('http://','https://')) else f"https://{u.strip()}" for u in urls_input.splitlines() if u.strip()]
    if len(urls) < 2:
        st.error('Au moins 2 URLs requises.')
        return

    # Scraping parallèle
    progress = st.progress(0)
    results: Dict[str,Dict] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_content_from_url, u): u for u in urls}
        for i, f in enumerate(as_completed(futures)):
            u = futures[f]
            raw,title,h1,subsecs = f.result()
            if raw and len(raw) > 100:
                results[u] = {'raw':raw,'title':title,'h1':h1,'subsecs':subsecs}
            else:
                st.warning(f"Contenu insuffisant: {u}")
            progress.progress((i+1)/len(urls))
    if len(results) < 2:
        st.error('Pas assez de contenus valides.')
        return

    # DataFrames
    df_stats = pd.DataFrame([{'URL':u,'Word Count':len(clean_text(v['raw']).split())} for u,v in results.items()])
    df_struct = pd.DataFrame([{'URL':u,'Title':v['title'],'H1':v['h1'],'Structure':'<br>'.join(v['subsecs'])} for u,v in results.items()])

    # Médiane et Moyenne
    median_wc = int(df_stats['Word Count'].median()); mean_wc = int(df_stats['Word Count'].mean())
    c1,c2 = st.columns(2)
    c1.metric('Médiane mots',median_wc)
    c2.metric('Moyenne mots',mean_wc)

        # Tableau fusionné: Structure + Nombre de mots
    df_comb = pd.DataFrame([{
        'URL': u,
        'Word Count': len(clean_text(v['raw']).split()),
        'Title': v['title'],
        'H1': v['h1'],
        'Structure': ' | '.join(v['subsecs'])
    } for u,v in results.items()])
    with st.expander('🗂️ Structure & Stats', expanded=True):
        st.write('Combinaison de la structure Hn et du nombre de mots pour chaque URL.')
        sty_comb = df_comb.style \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('white-space', 'normal')]}
            ]) \
            .set_properties(subset=['URL','Title','H1','Structure'], **{'text-align':'left', 'width':'250px'}) \
            .set_properties(subset=['Word Count'], **{'text-align':'center', 'width':'80px'})
        st.dataframe(sty_comb, use_container_width=True)

        # Expressions clés
    with st.expander('🧩 Expressions clés',expanded=True):
        st.markdown('_Expression | Mean Count | Doc Coverage (%)_')
        docs = [clean_text(v['raw']) for v in results.values()]
        stopw = stopwords.words(language)
        cv = CountVectorizer(ngram_range=(2,4), stop_words=stopw)
        X = cv.fit_transform(docs)
        terms = cv.get_feature_names_out()
        # Coverage in %
        cov = np.array((X>0).sum(axis=0)).ravel() / len(docs) * 100
        mask = cov >= 40
        rows = []
        for term in terms[mask][:40]:
            counts = X[:, cv.vocabulary_[term]].toarray().ravel()
            # mean only across docs containing term
            nonzero = counts[counts>0]
            mean_count = int(round(nonzero.mean())) if nonzero.size>0 else 0
            if is_relevant_expression(term):
                rows.append({
                    'Expression': term,
                    'Mean Count': mean_count,
                    'Doc Coverage (%)': int(round(cov[cv.vocabulary_[term]]))
                })
        df_cv = pd.DataFrame(rows).sort_values('Doc Coverage (%)', ascending=False)
        sty_cv = df_cv.style \
            .set_table_styles([
                {'selector':'th','props':[('text-align','center')]},
                {'selector':'td','props':[('white-space','normal')]}]) \
            .set_properties(subset=['Expression'], **{'text-align':'left', 'width':'200px'}) \
            .set_properties(subset=['Mean Count','Doc Coverage (%)'], **{'text-align':'center', 'width':'80px'})
        st.dataframe(sty_cv, use_container_width=True)

        # Readability Metrics
    with st.expander('📖 Readability Metrics',expanded=False):
        st.markdown(
            '_Flesch Ease_: + facile = + élevé<br>'+
            '_Kincaid Grade_: niveau scolaire<br>'+
            '_Gunning Fog_: + complexe = + élevé',unsafe_allow_html=True
        )
        # Calcul des scores pour chaque URL
        read = [{'URL': u, **get_readability_scores(v['raw'])} for u, v in results.items()]
        df_read = pd.DataFrame(read)
        # Moyennes globales
        mean_flesch = int(df_read['Flesch Ease'].mean())
        mean_kincaid = int(df_read['Kincaid Grade'].mean())
        mean_fog = int(df_read['Gunning Fog'].mean())
        col1, col2, col3 = st.columns(3)
        col1.metric('Moyenne Flesch Ease', mean_flesch)
        col2.metric('Moyenne Kincaid Grade', mean_kincaid)
        col3.metric('Moyenne Gunning Fog', mean_fog)
        # Affichage du tableau avec alignements
        sty_read = df_read.style \
            .set_table_styles([
                {'selector':'th','props':[('text-align','center')]},
                {'selector':'td','props':[('white-space','normal')]}]) \
            .set_properties(subset=['URL'], **{'text-align':'left', 'width':'250px'}) \
            .set_properties(subset=['Flesch Ease','Kincaid Grade','Gunning Fog'], **{'text-align':'center', 'width':'80px'})
        st.dataframe(sty_read, use_container_width=True)

        # Export
    st.download_button('📥 Stats CSV', df_stats.to_csv(index=False), file_name='stats.csv')
    st.download_button('📥 Struct CSV', df_comb.to_csv(index=False), file_name='structure_stats.csv')
    st.download_button('📥 Expr CSV', df_cv.to_csv(index=False), file_name='expressions.csv')
    st.download_button('📥 Readability CSV', pd.DataFrame(read).to_csv(index=False), file_name='readability.csv')
