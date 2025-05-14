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
from typing import Dict, List, Set, Tuple
from functools import lru_cache

# --- PDF Export Setup ---
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

# --- Logger Configuration ---
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- HTTP Session Setup ---
def get_session():
    """Get a reusable HTTP session with proper headers"""
    session = requests.Session()
    session.headers.update({'User-Agent': 'SemanticAnalyzer/1.0'})
    return session

# --- Regex Patterns (compile only once) ---
PATTERNS = {
    'punct': re.compile(r"[^\w\s]"),
    'digits': re.compile(r"\d+"),
    'spaces': re.compile(r"\s+")
}

# --- Excluded Expressions ---
EXCLUDED_EXPRESSIONS = {
    "bonjour", "merci", "au revoir", "salut", "bienvenue", "félicitations", "bravo",
    "cookies", "données personnelles", "caractère personnel", "protection des données", "mentions légales",
    "charte d'utilisation", "politique de confidentialité", "gérer les cookies", "stockées ou extraites",
    "gestion des cookies", "consentement aux cookies", "continuer sans accepter", "savoir plus", "en savoir plus",
    "utilisateur", "utilisateurs", "site web", "formulaire de contact", "the menu items", "avis", "blog",
    "guide d'achat", "newsletter", "rien à voir", "afin de", "valider votre inscription", "accéder au contenu",
    "page d'accueil", "prénom ou pseudo", "google llc", "envoyer des publicités", "adresse ip", "site", "email",
    "er", "css", "script", "footer", "header", "service client", "service spécifique"
}

# --- Download NLTK Resources ---
@st.cache_resource
def download_nltk_resources() -> None:
    """Download required NLTK resources if not already present"""
    for res in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f"{'tokenizers' if res=='punkt' else 'corpora'}/{res}")
        except LookupError:
            nltk.download(res)

# --- Text Processing Functions ---
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    txt = text.lower()
    txt = PATTERNS['punct'].sub(' ', txt)
    txt = PATTERNS['digits'].sub('', txt)
    txt = PATTERNS['spaces'].sub(' ', txt).strip()
    return txt

def is_relevant_expression(expr: str) -> bool:
    """Check if an expression is relevant (not generic/boilerplate)"""
    if not isinstance(expr, str) or len(expr.split()) < 2:
        return False
    return all(excl not in expr.lower() for excl in EXCLUDED_EXPRESSIONS)

# --- Web Scraping Functions ---
@st.cache_data(ttl=3600)
def extract_content_from_url(url: str) -> Dict:
    """Extract content and metadata from a URL"""
    data = {
        'raw': '',
        'title': '',
        'h1': '',
        'subsecs': [],
        'images': 0,
        'tables': 0,
        'buttons': 0,
        'internal': 0,
        'external': 0
    }
    
    try:
        session = get_session()
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        base = urlparse(url).netloc
        
        # Remove non-content elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'form']):
            tag.decompose()
            
        # Find main content container
        container = soup.find('main') or soup.find(id='content') or soup.body
        
        # Extract media counts
        data['images'] = len(container.find_all('img'))
        data['tables'] = len(container.find_all('table'))
        data['buttons'] = len(container.find_all('button'))
        
        # Extract links
        internal_links, external_links = set(), set()
        for a in container.find_all('a', href=True):
            href = a['href']
            parsed = urlparse(href)
            (external_links if parsed.netloc and parsed.netloc != base else internal_links).add(href)
            
        data['internal'] = len(internal_links)
        data['external'] = len(external_links)
        
        # Extract content
        paras = []
        for tag in container.find_all(['h1', 'h2', 'h3', 'p']):
            txt = tag.get_text(' ', strip=True)
            if tag.name == 'h1' and not data['h1']:
                data['h1'] = txt
            elif tag.name in ['h2', 'h3']:
                data['subsecs'].append(f"{tag.name.upper()}: {txt}")
            elif tag.name == 'p':
                paras.append(txt)
                
        data['raw'] = ' '.join(paras)
        data['title'] = soup.title.string.strip() if soup.title and soup.title.string else ''
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        
    return data

def get_readability_scores(text: str) -> Dict[str, int]:
    """Calculate readability scores for text"""
    if not text:
        return {'Flesch Ease': 0, 'Kincaid Grade': 0, 'Gunning Fog': 0}
    
    return {
        'Flesch Ease': int(round(textstat.flesch_reading_ease(text))),
        'Kincaid Grade': int(round(textstat.flesch_kincaid_grade(text))),
        'Gunning Fog': int(round(textstat.gunning_fog(text)))
    }

# --- Data Processing Functions ---
def process_content_vectors(docs: List[str], language: str) -> pd.DataFrame:
    """Process document vectors to extract common expressions"""
    if not docs:
        return pd.DataFrame(columns=['Expression', 'Mean Count', 'Coverage (%)'])
        
    stopw = stopwords.words(language)
    cv = CountVectorizer(ngram_range=(2, 4), stop_words=stopw)
    X = cv.fit_transform(docs)
    terms = cv.get_feature_names_out()
    cov = np.array((X > 0).sum(axis=0)).ravel() / len(docs) * 100
    
    rows = []
    for i, term in enumerate(terms):
        counts = X[:, cv.vocabulary_[term]].toarray().ravel()
        nz = counts[counts > 0]
        if nz.size > 0 and cov[i] >= 40 and is_relevant_expression(term):
            rows.append({
                'Expression': term,
                'Mean Count': int(round(nz.mean())),
                'Coverage (%)': int(round(cov[i]))
            })
            
    return pd.DataFrame(rows).query('`Mean Count` > 1 and `Coverage (%)` > 50') \
             .sort_values('Coverage (%)', ascending=False)

def create_dataframes(results: Dict, user_data: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create all dataframes needed for the analysis"""
    # Structure & Stats DataFrame
    df_struct = pd.DataFrame([
        {'URL': u, 'Title': d['title'], 'H1': d['h1'], 'Structure': ' | '.join(d['subsecs'])}
        for u, d in results.items()
    ])
    
    df_stats = pd.DataFrame([
        {'URL': u, 'Word Count': len(clean_text(d['raw']).split())}
        for u, d in results.items()
    ])
    
    # Media & Links DataFrame
    df_media = pd.DataFrame([
        {
            'URL': u,
            'Images': d['images'],
            'Tables': d['tables'],
            'Buttons': d['buttons'],
            'Internal': d['internal'],
            'External': d['external']
        }
        for u, d in results.items()
    ])
    
    # Readability DataFrame
    df_read = pd.DataFrame([
        {'URL': u, **get_readability_scores(d['raw'])}
        for u, d in results.items()
    ])
    
    return df_stats, df_struct, df_media, df_read

# --- UI Components ---
def display_metrics(df_read: pd.DataFrame) -> None:
    """Display average readability metrics"""
    mean_vals = df_read.mean(numeric_only=True).round().astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric('Moy. Flesch Ease', mean_vals['Flesch Ease'])
    c2.metric('Moy. Kincaid Grade', mean_vals['Kincaid Grade'])
    c3.metric('Moy. Gunning Fog', mean_vals['Gunning Fog'])

def display_comparison(user_data: Dict, df_stats: pd.DataFrame, df_media: pd.DataFrame, 
                      df_read: pd.DataFrame, terms_group: List[str]) -> None:
    """Display comparison between user page and competitors"""
    if not user_data:
        return
        
    st.subheader('🔍 Analyse comparative de votre page')

    # Word Count
    uwc = len(clean_text(user_data['raw']).split())
    median_wc = int(df_stats['Word Count'].median())
    mean_wc = int(df_stats['Word Count'].mean())
    mcol1, mcol2 = st.columns(2)
    mcol1.metric('Mots (votre page)', uwc, delta=int(uwc - mean_wc))
    mcol2.metric('Médiane groupe', median_wc)

    # Missing Keywords
    missing = [t for t in terms_group if t not in clean_text(user_data['raw'])]
    if missing:
        st.table(pd.DataFrame({'Mots clés manquants': missing}))
    else:
        st.write('Aucun mot clé manquant pertinent.')

    # Media & Links Comparison
    media_mean = df_media[['Images', 'Internal', 'External']].mean().round().astype(int)
    imgs, intern, extern = user_data['images'], user_data['internal'], user_data['external']
    col_img, col_int, col_ext = st.columns(3)
    col_img.metric('Images (vous)', imgs, delta=int(imgs - media_mean['Images']))
    col_int.metric('Liens internes (vous)', intern, delta=int(intern - media_mean['Internal']))
    col_ext.metric('Liens externes (vous)', extern, delta=int(extern - media_mean['External']))

    # Readability Comparison
    ur = get_readability_scores(user_data['raw'])
    mean_read = df_read.mean(numeric_only=True).round().astype(int)
    r1, r2, r3 = st.columns(3)
    r1.metric('Flesch Ease (vous)', ur['Flesch Ease'], delta=int(ur['Flesch Ease'] - mean_read['Flesch Ease']))
    r2.metric('Kincaid Grade (vous)', ur['Kincaid Grade'], delta=int(ur['Kincaid Grade'] - mean_read['Kincaid Grade']))
    r3.metric('Gunning Fog (vous)', ur['Gunning Fog'], delta=int(ur['Gunning Fog'] - mean_read['Gunning Fog']))

def export_options(df_stats: pd.DataFrame, df_struct: pd.DataFrame, df_media: pd.DataFrame, 
                  df_cv: pd.DataFrame, df_read: pd.DataFrame, user_data: Dict = None) -> None:
    """Display export options"""
    st.markdown('---')
    
    # CSV exports
    merged_stats = pd.merge(df_stats, df_struct, on='URL')
    st.download_button('📥 Export CSV Structure & Stats', merged_stats.to_csv(index=False), 
                      file_name='structure_stats.csv')
    st.download_button('📥 Export CSV Media & Links', df_media.to_csv(index=False), 
                      file_name='media_links.csv')
    st.download_button('📥 Export CSV Expressions clés', df_cv.to_csv(index=False), 
                      file_name='expressions.csv')
    st.download_button('📥 Export CSV Readability', df_read.to_csv(index=False), 
                      file_name='readability.csv')

    # PDF export
    if PDFKIT_AVAILABLE:
        html = '<html><head><meta charset="utf-8"></head><body>'
        html += '<h1>📄 Rapport Semantic Analyzer</h1>'
        
        # Add sections
        sections = [
            ('Structure & Stats', merged_stats),
            ('Media & Links', df_media),
            ('Expressions clés', df_cv),
            ('Readability Metrics', df_read)
        ]
        
        # Add user data comparison if available
        if user_data:
            media_mean = df_media[['Images', 'Internal', 'External']].mean().round().astype(int)
            imgs, intern, extern = user_data['images'], user_data['internal'], user_data['external']
            
            comp_df = pd.DataFrame({
                'Metric': ['Images', 'Liens internes', 'Liens externes'],
                'Vous': [imgs, intern, extern],
                'Moyenne': [media_mean['Images'], media_mean['Internal'], media_mean['External']],
                'Delta': [imgs - media_mean['Images'], intern - media_mean['Internal'], 
                          extern - media_mean['External']]
            })
            sections.append(('Comparative de votre page', comp_df))
            
        for title, df in sections:
            html += f'<h2>{title}</h2>' + df.to_html(index=False)
            
        html += '</body></html>'
        
        try:
            pdf = pdfkit.from_string(html, False)
            st.download_button(
                '📥 Télécharger PDF complet',
                data=pdf,
                file_name='rapport_semantic_analyzer.pdf',
                mime='application/pdf'
            )
        except OSError:
            st.warning('wkhtmltopdf non trouvé : installez wkhtmltopdf pour activer l\'export PDF.')
    else:
        st.warning('Pour exporter en PDF, installez pdfkit et wkhtmltopdf.')

# --- Main App Function ---
def run() -> None:
    """Main application function"""
    st.title('🔍 Semantic Analyzer')
    st.markdown('**Comparez plusieurs pages et comparez-les à votre page**')

    # Download NLTK resources
    download_nltk_resources()

    # Inputs
    urls_input = st.text_area('Entrez les URLs à comparer (une par ligne)', height=150)
    user_url = st.text_input('Entrez votre URL pour comparaison (facultatif)')
    language = st.selectbox('Langue du contenu', ['french', 'english'])

    if not st.button('Analyser'):
        return
        
    # Process input URLs
    urls = [u.strip() if u.strip().startswith(('http://', 'https://')) else f"https://{u.strip()}"
            for u in urls_input.splitlines() if u.strip()]
            
    if len(urls) < 2:
        st.error('Veuillez fournir au moins 2 URLs.')
        return

    # Scrape competitor pages in parallel
    progress = st.progress(0)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(extract_content_from_url, url): url for url in urls}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            results[url] = future.result()
            progress.progress((i + 1) / len(urls))

    # Scrape user page if provided
    user_data = None
    if user_url:
        if not user_url.startswith(('http://', 'https://')):
            user_url = 'https://' + user_url.strip()
        user_data = extract_content_from_url(user_url)

    # Create DataFrames
    df_stats, df_struct, df_media, df_read = create_dataframes(results)
    
    # Process content vectors
    docs = [clean_text(d['raw']) for d in results.values()]
    df_cv = process_content_vectors(docs, language)

    # Display results
    with st.expander('🗂️ Structure & Stats', expanded=True):
        st.dataframe(pd.merge(df_stats, df_struct, on='URL'), use_container_width=True)

    with st.expander('📊 Media & Links', expanded=False):
        st.dataframe(df_media, use_container_width=True)

    with st.expander('🧩 Expressions clés', expanded=False):
        st.dataframe(df_cv, use_container_width=True)

    # Display metrics and comparisons
    display_metrics(df_read)
    
    with st.expander('📖 Readability Metrics', expanded=False):
        st.dataframe(df_read, use_container_width=True)

    # Display comparative summary if user URL was provided
    if user_data:
        display_comparison(user_data, df_stats, df_media, df_read, df_cv['Expression'].tolist())

    # Display export options
    export_options(df_stats, df_struct, df_media, df_cv, df_read, user_data)

if __name__ == '__main__':
    run()