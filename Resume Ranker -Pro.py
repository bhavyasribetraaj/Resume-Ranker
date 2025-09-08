import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import zipfile
import tempfile
from io import BytesIO, StringIO
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

try:
    import fitz
except Exception:
    fitz = None
try:
    import docx
except Exception:
    docx = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
try:
    from sentence_transformers import SentenceTransformer
    _EMB_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _EMB_AVAILABLE = False
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize, sent_tokenize
    nltk_available = True
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet")
except Exception:
    nltk_available = False
try:
    from fpdf import FPDF
    _FPDF = True
except Exception:
    _FPDF = False

st.set_page_config(page_title="Resume Ranker — Pro+", layout="wide", initial_sidebar_state="expanded")

EN_STOP = set(ENGLISH_STOP_WORDS)

# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "compare_candidates" not in st.session_state:
    st.session_state.compare_candidates = []
if "jd_keywords" not in st.session_state:
    st.session_state.jd_keywords = []
if "jd_sector" not in st.session_state:
    st.session_state.jd_sector = ""
if "analysis_timestamp" not in st.session_state:
    st.session_state.analysis_timestamp = None

def read_pdf_bytes(b: bytes):
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\n".join([p.get_text() for p in doc])
    except:
        return ""

def read_docx_bytes(b: bytes):
    if docx is None:
        return ""
    try:
        tmp = BytesIO(b)
        document = docx.Document(tmp)
        return "\n".join([p.text for p in document.paragraphs])
    except:
        return ""

def read_txt_bytes(b: bytes):
    try:
        return b.decode(errors="ignore")
    except:
        return str(b)

def read_html_bytes(b: bytes):
    if BeautifulSoup is None:
        return ""
    try:
        soup = BeautifulSoup(b, "html.parser")
        return soup.get_text(separator=" ")
    except:
        return ""

def process_tabular_bytes(b: bytes, ext: str):
    try:
        if ext == ".csv":
            df = pd.read_csv(BytesIO(b))
        else:
            df = pd.read_excel(BytesIO(b))
        texts = []
        if "text" in df.columns:
            texts = [str(r) for r in df["text"].fillna("")]
        elif "resume" in df.columns:
            texts = [str(r) for r in df["resume"].fillna("")]
        else:
            texts = [" ".join(map(str, row)) for row in df.values]
        return texts
    except:
        return []

def parse_uploaded_file(uf):
    name = uf.name
    ext = os.path.splitext(name)[1].lower()
    b = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
    if ext == ".pdf":
        return [read_pdf_bytes(b)]
    if ext == ".docx":
        return [read_docx_bytes(b)]
    if ext == ".txt":
        return [read_txt_bytes(b)]
    if ext in [".html", ".htm"]:
        return [read_html_bytes(b)]
    if ext in [".csv", ".xlsx", ".xls"]:
        return process_tabular_bytes(b, ext)
    if ext == ".zip":
        texts = []
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "tmp.zip")
            open(zpath, "wb").write(b)
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(td)
            for root, _, files in os.walk(td):
                for f in files:
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        fake = type("F", (), {"name": f, "getvalue": fh.read()})
                        texts.extend(parse_uploaded_file(fake))
        return texts
    return []

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text)
    return t.strip()

def tokenize_and_filter(text, lemmatize=False):
    text = basic_clean(text).lower()
    tokens = re.findall(r"\b[a-zA-Z0-9\+\#\.\-]+\b", text)
    tokens = [t for t in tokens if t not in EN_STOP and len(t) > 1]
    if lemmatize and nltk_available:
        try:
            lem = [WordNetLemmatizer().lemmatize(t) for t in tokens]
            return lem
        except:
            return tokens
    return tokens

def extract_keywords_tfidf(corpus, top_n=12):
    try:
        v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
        X = v.fit_transform(corpus)
        feature_array = np.array(v.get_feature_names_out())
        tfidf_sorting = np.argsort(X.toarray().sum(axis=0))[::-1]
        top_n = min(top_n, len(feature_array))
        top_feats = feature_array[tfidf_sorting][:top_n]
        return [str(x) for x in top_feats]
    except:
        words = " ".join(corpus).split()
        freqs = {}
        for w in words:
            wl = w.lower()
            if wl not in EN_STOP and len(wl) > 2:
                freqs[wl] = freqs.get(wl, 0) + 1
        sorted_w = sorted(freqs.items(), key=lambda x:-x[1])
        return [w for w,_ in sorted_w[:top_n]]

@st.cache_resource
def load_embedding_model(name="all-MiniLM-L6-v2"):
    if not _EMB_AVAILABLE:
        return None
    try:
        return SentenceTransformer(name)
    except:
        return None

emb_model = load_embedding_model()

def compute_embeddings(texts):
    if emb_model is None:
        return None
    try:
        return np.array(emb_model.encode(texts, normalize_embeddings=True))
    except:
        return None

def compute_tfidf_sim(jd, resumes):
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
    M = v.fit_transform([jd] + resumes)
    sims = cosine_similarity(M[0:1], M[1:]).flatten()
    return sims

def sector_detect_simple(text):
    text_l = text.lower()
    sectors = {
        "Law": ["law", "court", "advocate", "litigation", "bar council", "case", "advocacy", "legal", "attorney"],
        "Finance": ["finance", "accounting", "auditor", "gaap", "ifrs", "financial", "banking", "investment", "cfa"],
        "Tech": ["software", "developer", "engineer", "python", "java", "backend", "frontend", "coding", "programming"],
        "HR": ["recruit", "talent", "hr", "human resources", "recruiting", "hiring"],
        "Healthcare": ["medical", "healthcare", "doctor", "nurse", "hospital", "clinical", "patient"],
        "Marketing": ["marketing", "digital", "seo", "campaign", "brand", "advertising", "social media"],
        "Sales": ["sales", "business development", "account", "revenue", "client", "customer"],
        "Education": ["teacher", "education", "instructor", "curriculum", "student", "academic"]
    }
    
    scores = {}
    for sector, keywords in sectors.items():
        scores[sector] = sum(1 for k in keywords if k in text_l)
    
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "General"
    return best

def extract_experience(text):
    text = text.lower()
    patterns = [
        r"(\d{1,2}\+?)\s*(?:years|year|yrs|yr)\b",
        r"(\d{1,2})\s*\-\s*(\d{1,2})\s*(?:years|year|yrs|yr)\b",
        r"(\d{1,2})\s*(?:\+)?\s*years\b"
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text):
            groups = m.groups()
            for g in groups:
                if g and g.isdigit():
                    found.append(int(g))
                elif g and re.match(r"\d+\+", g):
                    found.append(int(g.replace("+","")))
    if not found:
        return None
    return max(found)

def extract_education_level(text):
    """Extract highest education level mentioned"""
    text = text.lower()
    levels = {
        "PhD": ["phd", "ph.d", "doctorate", "doctoral"],
        "Masters": ["masters", "master's", "mba", "m.s", "m.a", "m.tech", "m.sc"],
        "Bachelors": ["bachelor", "b.s", "b.a", "b.tech", "b.sc", "b.e", "undergraduate"],
        "Associate": ["associate", "diploma"],
        "High School": ["high school", "secondary", "12th", "grade 12"]
    }
    
    for level, keywords in levels.items():
        if any(kw in text for kw in keywords):
            return level
    return "Not specified"

def extract_skills(text):
    """Extract technical skills from resume"""
    text = text.lower()
    skill_categories = {
        "Programming": ["python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "kotlin", "swift"],
        "Web Tech": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "spring"],
        "Databases": ["sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis"],
        "Cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins"],
        "Analytics": ["excel", "tableau", "power bi", "r", "stata", "spss"],
        "Design": ["photoshop", "illustrator", "figma", "sketch", "adobe"]
    }
    
    found_skills = {}
    for category, skills in skill_categories.items():
        found = [skill for skill in skills if skill in text]
        if found:
            found_skills[category] = found
    
    return found_skills

def calculate_resume_quality_score(text):
    """Calculate resume quality based on various factors"""
    score = 0
    text_lower = text.lower()
    
    # Length check (optimal range)
    word_count = len(text.split())
    if 300 <= word_count <= 800:
        score += 20
    elif 200 <= word_count <= 1000:
        score += 10
    
    # Contact info
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        score += 15
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
        score += 10
    
    # Professional sections
    sections = ["experience", "education", "skills", "projects", "summary"]
    section_score = sum(10 for section in sections if section in text_lower)
    score += min(section_score, 30)
    
    # Action verbs
    action_verbs = ["managed", "led", "developed", "created", "implemented", "achieved", "improved"]
    verb_score = sum(3 for verb in action_verbs if verb in text_lower)
    score += min(verb_score, 15)
    
    # Quantified achievements (numbers/percentages)
    if re.search(r'\d+%|\$\d+|\d+\s*(million|thousand|k\b)', text_lower):
        score += 10
    
    return min(score, 100)

def summarize_text(text, top_n=3):
    if not text or len(text.split()) < 30:
        return text.strip()
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= top_n:
        return " ".join(sents)
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    try:
        M = v.fit_transform(sents)
        ranks = np.asarray(M.sum(axis=1)).ravel()
        top_idx = ranks.argsort()[::-1][:top_n]
        top_idx_sorted = sorted(top_idx)
        return " ".join([sents[i] for i in top_idx_sorted])
    except:
        return " ".join(sents[:top_n])

def highlight_text(text, keywords):
    if not keywords:
        return basic_clean(text)
    words = [k for k in keywords if k and len(k.strip())>0]
    if not words:
        return basic_clean(text)
    esc = [re.escape(w) for w in sorted(set(words), key=lambda x:-len(x))]
    pattern = r"\b(" + "|".join(esc) + r")\b"
    try:
        return re.sub(pattern, lambda m: f"<mark style='background:#ffd54f'>{m.group(0)}</mark>", text, flags=re.IGNORECASE)
    except:
        return basic_clean(text)

def create_radar_chart(scores_dict, title):
    """Create radar chart for skill comparison"""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title=title,
        template="plotly_dark"
    )
    return fig

def cluster_candidates(df, n_clusters=3):
    """Cluster candidates based on their scores"""
    if len(df) < n_clusters:
        return df
    
    features = df[['Score', 'Semantic(%)', 'Coverage(%)']].values
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)
        cluster_names = {0: 'Top Tier', 1: 'Mid Tier', 2: 'Entry Level'}
        df['Tier'] = df['Cluster'].map(cluster_names)
        return df
    except:
        return df

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem;
}
.candidate-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    border-left: 5px solid #00d4aa;
}
.top-candidate {
    border-left-color: #ffd700 !important;
    background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
}
.stMetric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Main App
st.markdown('<h1 class="main-header">🚀 Resume Ranker — Pro+</h1>', unsafe_allow_html=True)
st.markdown("### Advanced AI-Powered Resume Analysis & Ranking System")

with st.sidebar:
    st.header("🎛️ Configuration Panel")
    method = st.selectbox("🧠 AI Method", ["Embeddings if available", "Force TF-IDF"], index=0)
    keyword_toggle = st.checkbox("🔍 Show matched keywords & highlights", value=True)
    lemmatize_toggle = st.checkbox("📝 Enable lemmatization", value=True)
    weight_semantic = st.slider("⚖️ Semantic weight in final score", 0.0, 1.0, 0.7, 0.05)
    top_keywords_n = st.slider("🔑 Extract top N JD keywords", 3, 20, 10)
    
    st.markdown("#### 🎯 Filtering Options")
    min_score_filter = st.slider("📊 Show only candidates >= score (%)", 0, 100, 0)
    search_skill = st.text_input("🔎 Search by skill/keyword")
    min_experience = st.slider("💼 Minimum experience (years)", 0, 20, 0)
    education_filter = st.selectbox("🎓 Education level", 
                                   ["All", "PhD", "Masters", "Bachelors", "Associate", "High School"])
    
    st.markdown("#### 📈 Advanced Features")
    show_clustering = st.checkbox("🎯 Enable candidate clustering", value=True)
    show_quality_score = st.checkbox("⭐ Show resume quality scores", value=True)
    show_skills_analysis = st.checkbox("🛠️ Advanced skills analysis", value=True)
    
    if st.button("🗑️ Clear Analysis", type="secondary"):
        st.session_state.analysis_complete = False
        st.session_state.results_df = None
        st.session_state.compare_candidates = []
        st.rerun()

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("📁 Upload resumes (PDF/DOCX/TXT/HTML/CSV/XLSX/ZIP)", 
                               accept_multiple_files=True, 
                               help="Supports multiple formats and batch uploads via ZIP")
with col2:
    jd_mode = st.radio("📋 Job Description Input", ["Paste", "Upload"], index=0)
    if jd_mode == "Paste":
        jd_text = st.text_area("✍️ Paste job description here", height=260)
    else:
        jd_file = st.file_uploader("📤 Upload JD file", type=["pdf","docx","txt","html"])
        jd_text = ""
        if jd_file:
            jtexts = parse_uploaded_file(jd_file)
            jd_text = jtexts[0] if jtexts else ""

analyze = st.button("🚀 Analyze & Rank Candidates", type="primary", use_container_width=True)

if analyze and not st.session_state.analysis_complete:
    if not uploaded:
        st.warning("⚠️ Please upload at least one resume file.")
        st.stop()
    if not jd_text or not jd_text.strip():
        st.warning("⚠️ Please provide a job description (paste or upload).")
        st.stop()

    with st.spinner("🔄 Processing resumes and analyzing candidates..."):
        raw_texts = []
        raw_names = []
        for uf in uploaded:
            parts = parse_uploaded_file(uf)
            if not parts:
                continue
            for i, t in enumerate(parts):
                base = os.path.splitext(uf.name)[0]
                name = f"{base}_{i+1}" if len(parts) > 1 else base
                raw_names.append(name)
                raw_texts.append(basic_clean(t))

        if not raw_texts:
            st.error("❌ No readable text extracted from uploaded files.")
            st.stop()

        jd_clean = basic_clean(jd_text)
        st.session_state.jd_sector = sector_detect_simple(jd_clean)
        st.session_state.jd_keywords = extract_keywords_tfidf([jd_clean], top_n=top_keywords_n)
        
        if not st.session_state.jd_keywords:
            st.session_state.jd_keywords = extract_keywords_tfidf([jd_clean] + raw_texts, top_n=top_keywords_n)

        # Compute similarities
        if method == "Embeddings if available" and emb_model is not None:
            embs = compute_embeddings([jd_clean] + raw_texts)
            if embs is not None:
                jd_emb = embs[0:1]
                res_embs = embs[1:]
                sem_sims = cosine_similarity(jd_emb, res_embs).flatten()
            else:
                sem_sims = compute_tfidf_sim(jd_clean, raw_texts)
        else:
            sem_sims = compute_tfidf_sim(jd_clean, raw_texts)

        tokenized_jd = tokenize_and_filter(jd_clean, lemmatize=(lemmatize_toggle and nltk_available))
        tokenized_jd_set = set(tokenized_jd + [k.lower() for k in st.session_state.jd_keywords])

        # Process each resume
        matched_list = []
        missing_list = []
        coverage = []
        experiences = []
        education_levels = []
        summaries = []
        skills_analysis = []
        quality_scores = []
        
        for txt in raw_texts:
            tokens = tokenize_and_filter(txt, lemmatize=(lemmatize_toggle and nltk_available))
            tset = set(tokens)
            matched = sorted(list(tokenized_jd_set.intersection(tset)))
            missing = sorted(list(tokenized_jd_set - tset))
            matched_list.append(matched)
            missing_list.append(missing)
            coverage.append(len(matched) / max(1, len(tokenized_jd_set)))
            experiences.append(extract_experience(txt))
            education_levels.append(extract_education_level(txt))
            summaries.append(summarize_text(txt, top_n=3))
            
            if show_skills_analysis:
                skills_analysis.append(extract_skills(txt))
            else:
                skills_analysis.append({})
                
            if show_quality_score:
                quality_scores.append(calculate_resume_quality_score(txt))
            else:
                quality_scores.append(0)

        # Calculate final scores
        sem_arr = np.array(sem_sims)
        sem_norm = (sem_arr - sem_arr.min()) / (sem_arr.max() - sem_arr.min() + 1e-9)
        cov_arr = np.array(coverage)
        cov_norm = (cov_arr - cov_arr.min()) / (cov_arr.max() - cov_arr.min() + 1e-9)
        final_scores = weight_semantic * sem_norm + (1 - weight_semantic) * cov_norm
        final_pct = (final_scores * 100).round(1)

        # Create results DataFrame
        df = pd.DataFrame({
            "Candidate": raw_names,
            "Score": final_scores,
            "Score(%)": final_pct,
            "Semantic(%)": (sem_norm * 100).round(1),
            "Coverage(%)": (cov_norm * 100).round(1),
            "Quality Score": quality_scores,
            "Matched": [", ".join(m[:30]) if m else "-" for m in matched_list],
            "Missing": [", ".join(m[:30]) if m else "-" for m in missing_list],
            "Experience": [f"{e} yrs" if e is not None else "-" for e in experiences],
            "Experience_Numeric": [e if e is not None else 0 for e in experiences],
            "Education": education_levels,
            "Skills": skills_analysis,
            "Summary": summaries,
            "RawText": raw_texts
        })
        
        df.sort_values(by="Score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Add clustering
        if show_clustering and len(df) >= 3:
            df = cluster_candidates(df)
        
        st.session_state.results_df = df
        st.session_state.analysis_complete = True
        st.session_state.analysis_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    st.success("✅ Analysis completed successfully!")
    st.rerun()

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.results_df is not None:
    df = st.session_state.results_df.copy()
    
    # Apply filters
    if search_skill:
        search_lower = search_skill.lower()
        df = df[df["RawText"].str.lower().str.contains(re.escape(search_lower)) | 
                df["Matched"].str.lower().str.contains(re.escape(search_lower))].reset_index(drop=True)
    
    if min_experience > 0:
        df = df[df["Experience_Numeric"] >= min_experience].reset_index(drop=True)
    
    if education_filter != "All":
        df = df[df["Education"] == education_filter].reset_index(drop=True)
    
    df_display = df[df["Score(%)"] >= min_score_filter].reset_index(drop=True)
    
    # Display metrics
    total = len(st.session_state.results_df)
    filtered = len(df_display)
    best = df.iloc[0] if len(df) > 0 else None
    avg_score = df["Score(%)"].mean().round(1) if len(df) > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Total Resumes", total)
    col2.metric("🎯 After Filters", filtered)
    col3.metric("🏆 Top Candidate", best["Candidate"] if best is not None else "-")
    col4.metric("⭐ Top Score", f"{best['Score(%)']}%" if best is not None else "-")
    col5.metric("📈 Avg Score", f"{avg_score}%")
    
    # Analysis timestamp and JD info
    if st.session_state.analysis_timestamp:
        st.info(f"📅 Analysis completed on: {st.session_state.analysis_timestamp} | "
                f"🏢 Detected sector: {st.session_state.jd_sector}")
    
    # Main content layout
    left_col, right_col = st.columns([2.2, 1.8])
    
    with left_col:
        st.markdown("### 🏆 Ranked Candidates")
        
        # Display candidates
        for idx, row in df_display.iterrows():
            score = float(row["Score(%)"])
            quality = row["Quality Score"] if show_quality_score else 0
            
            # Determine styling based on score
            if score >= 80:
                label, color, emoji = "Excellent", "#00ff88", "🌟"
            elif score >= 65:
                label, color, emoji = "Strong", "#00d4aa", "💪"
            elif score >= 50:
                label, color, emoji = "Moderate", "#ffa726", "👍"
            else:
                label, color, emoji = "Needs Review", "#ff5722", "⚠️"
            
            # Card styling
            tier_badge = ""
            if show_clustering and "Tier" in row:
                tier_colors = {"Top Tier": "#ffd700", "Mid Tier": "#c0c0c0", "Entry Level": "#cd7f32"}
                tier_color = tier_colors.get(row["Tier"], "#666")
                tier_badge = f"<span style='background:{tier_color};color:black;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:bold;margin-left:8px'>{row['Tier']}</span>"
            
            card_html = f"""
            <div style='background:linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding:16px; border-radius:15px; margin:8px 0; 
                        border-left:5px solid {color}; 
                        {"box-shadow: 0 0 20px rgba(255,215,0,0.3);" if idx == 0 else ""}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-size:20px; color:white; font-weight:700;'>
                            {emoji} #{idx+1}. {row['Candidate']}
                        </span>
                        {tier_badge}
                        <div style='color:#b0c4de; font-size:13px; margin-top:4px;'>
                            🎯 Semantic: {row['Semantic(%)']}% • 📋 Coverage: {row['Coverage(%)']}% • 
                            💼 Exp: {row['Experience']} • 🎓 {row['Education']}
                            {f" • ⭐ Quality: {quality}%" if show_quality_score else ""}
                        </div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='background:{color}; color:white; padding:8px 12px; 
                                    border-radius:8px; font-weight:700; margin-bottom:4px;'>
                            {label}
                        </div>
                        <div style='font-size:16px; color:white;'>{row['Score(%)']}%</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Expandable details
            with st.expander(f"📋 Detailed Analysis — {row['Candidate']}"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write(f"**🎯 Final Score:** {row['Score(%)']}%")
                    st.write(f"**🧠 Semantic Similarity:** {row['Semantic(%)']}%")
                    st.write(f"**📊 Keyword Coverage:** {row['Coverage(%)']}%")
                    if show_quality_score:
                        st.write(f"**⭐ Resume Quality:** {row['Quality Score']}%")
                    st.write(f"**💼 Experience:** {row['Experience']}")
                    st.write(f"**🎓 Education:** {row['Education']}")
                
                with detail_col2:
                    if show_skills_analysis and row["Skills"]:
                        st.write("**🛠️ Technical Skills Found:**")
                        for category, skills in row["Skills"].items():
                            st.write(f"• {category}: {', '.join(skills)}")
                    
                    if show_clustering and "Tier" in row:
                        st.write(f"**🎯 Candidate Tier:** {row['Tier']}")
                
                st.write("**📝 Summary:**")
                st.write(row["Summary"])
                
                if keyword_toggle:
                    st.write("**✅ Matched Keywords:**", row["Matched"] if row["Matched"] != "-" else "None")
                    st.write("**❌ Missing Keywords:**", row["Missing"] if row["Missing"] != "-" else "None")
                
                # Highlighted resume text
                if keyword_toggle and row["Matched"] != "-":
                    st.write("**📄 Resume with Highlighted Keywords:**")
                    highlighted = highlight_text(row["RawText"], [k.strip() for k in row["Matched"].split(",") if k.strip()])
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.text_area("📄 Full Resume Text", row["RawText"], height=200)
    
    with right_col:
        st.markdown("### 📊 Visual Analytics")
        
        if len(df_display) > 0:
            # Score distribution chart
            fig_scores = px.histogram(df_display, x="Score(%)", nbins=10, 
                                    title="📈 Score Distribution",
                                    color_discrete_sequence=["#00d4aa"])
            fig_scores.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Horizontal bar chart
            fig_bar = px.bar(df_display.head(10), y="Candidate", x="Score(%)", 
                           orientation="h", title="🏆 Top Candidates",
                           color="Score(%)", color_continuous_scale="Viridis")
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Experience vs Score scatter
            if show_quality_score:
                fig_scatter = px.scatter(df_display, x="Experience_Numeric", y="Score(%)", 
                                       size="Quality Score", hover_name="Candidate",
                                       title="💼 Experience vs Score vs Quality",
                                       color="Score(%)", color_continuous_scale="Plasma")
                fig_scatter.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Clustering visualization
            if show_clustering and "Tier" in df_display.columns:
                fig_cluster = px.scatter(df_display, x="Semantic(%)", y="Coverage(%)", 
                                       color="Tier", size="Score(%)", hover_name="Candidate",
                                       title="🎯 Candidate Clustering")
                fig_cluster.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_cluster, use_container_width=True)
            
        else:
            st.info("📊 No candidates to visualize after applying filters.")
        
        # JD Keywords
        if st.session_state.jd_keywords:
            st.markdown("#### 🔑 Key Job Requirements")
            keywords_df = pd.DataFrame({
                "Keyword": st.session_state.jd_keywords[:8], 
                "Importance": range(len(st.session_state.jd_keywords[:8]), 0, -1)
            })
            fig_keywords = px.bar(keywords_df, x="Importance", y="Keyword", 
                                orientation="h", color="Importance",
                                color_continuous_scale="Blues")
            fig_keywords.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_keywords, use_container_width=True)
    
    # Advanced Analytics Section
    st.markdown("---")
    st.markdown("### 🔬 Advanced Analytics & Insights")
    
    analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
    
    with analytics_col1:
        if show_skills_analysis:
            # Skills analysis
            all_skills = {}
            for _, row in df_display.iterrows():
                for category, skills in row["Skills"].items():
                    if category not in all_skills:
                        all_skills[category] = set()
                    all_skills[category].update(skills)
            
            if all_skills:
                st.markdown("#### 🛠️ Skills Landscape")
                skills_data = []
                for category, skills in all_skills.items():
                    skills_data.append({"Category": category, "Count": len(skills)})
                
                if skills_data:
                    skills_df = pd.DataFrame(skills_data)
                    fig_skills = px.pie(skills_df, values="Count", names="Category", 
                                      title="Skill Categories Distribution")
                    fig_skills.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_skills, use_container_width=True)
    
    with analytics_col2:
        # Education distribution
        if len(df_display) > 0:
            education_counts = df_display["Education"].value_counts()
            fig_education = px.pie(values=education_counts.values, names=education_counts.index,
                                 title="🎓 Education Distribution")
            fig_education.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_education, use_container_width=True)
    
    with analytics_col3:
        # Experience distribution
        if len(df_display) > 0:
            exp_ranges = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]
            exp_counts = [0, 0, 0, 0]
            for exp in df_display["Experience_Numeric"]:
                if exp <= 2:
                    exp_counts[0] += 1
                elif exp <= 5:
                    exp_counts[1] += 1
                elif exp <= 10:
                    exp_counts[2] += 1
                else:
                    exp_counts[3] += 1
            
            fig_exp = px.bar(x=exp_ranges, y=exp_counts, title="💼 Experience Distribution",
                           color=exp_counts, color_continuous_scale="Turbo")
            fig_exp.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_exp, use_container_width=True)
    
    # Comparison Section
    st.markdown("---")
    st.markdown("### 🔍 Candidate Comparison & Export Tools")
    
    comparison_col, export_col = st.columns([2, 1])
    
    with comparison_col:
        st.markdown("#### 👥 Side-by-Side Comparison")
        candidates_list = df["Candidate"].tolist()
        selected_candidates = st.multiselect(
            "Select up to 4 candidates to compare",
            candidates_list,
            default=st.session_state.compare_candidates[:3],
            max_selections=4,
            key="candidate_selector"
        )
        
        # Update session state
        st.session_state.compare_candidates = selected_candidates
        
        if selected_candidates:
            comparison_df = df[df["Candidate"].isin(selected_candidates)].reset_index(drop=True)
            
            # Create comparison table
            comparison_metrics = comparison_df[["Candidate", "Score(%)", "Semantic(%)", "Coverage(%)", 
                                             "Experience", "Education"]]
            if show_quality_score:
                comparison_metrics = pd.concat([comparison_metrics, comparison_df[["Quality Score"]]], axis=1)
            
            st.dataframe(comparison_metrics, use_container_width=True)
            
            # Comparison radar chart
            if len(selected_candidates) <= 3:
                fig_radar = go.Figure()
                
                for _, row in comparison_df.iterrows():
                    metrics = ['Score(%)', 'Semantic(%)', 'Coverage(%)']
                    values = [row[m] for m in metrics]
                    if show_quality_score:
                        metrics.append('Quality Score')
                        values.append(row['Quality Score'])
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=row['Candidate']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title="📊 Multi-Candidate Comparison",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    with export_col:
        st.markdown("#### 📥 Export & Reports")
        
        # CSV Export
        csv_data = df.drop(['RawText'], axis=1).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Results (CSV)",
            data=csv_data,
            file_name=f"resume_analysis_{st.session_state.analysis_timestamp.replace(' ', '_').replace(':', '-')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary Report
        if st.button("📋 Generate Summary Report", use_container_width=True):
            summary_text = f"""
# Resume Analysis Summary Report
Generated on: {st.session_state.analysis_timestamp}

## Job Description Analysis
- Detected Sector: {st.session_state.jd_sector}
- Key Requirements: {', '.join(st.session_state.jd_keywords[:10])}

## Candidate Overview
- Total Candidates: {len(st.session_state.results_df)}
- Average Score: {df['Score(%)'].mean():.1f}%
- Top Candidate: {df.iloc[0]['Candidate']} ({df.iloc[0]['Score(%)']:.1f}%)

## Top 5 Candidates:
"""
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                summary_text += f"{i+1}. {row['Candidate']} - {row['Score(%)']:.1f}% (Experience: {row['Experience']})\n"
            
            st.download_button(
                label="📄 Download Summary (TXT)",
                data=summary_text.encode('utf-8'),
                file_name="analysis_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Individual PDF Reports
        if _FPDF and len(df) > 0:
            st.markdown("**Individual Reports:**")
            selected_candidate = st.selectbox("Select candidate", df["Candidate"].tolist())
            
            if st.button("📄 Generate PDF Report", use_container_width=True):
                candidate_row = df[df["Candidate"] == selected_candidate].iloc[0]
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(0, 10, f"Candidate Analysis Report", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.ln(10)
                
                # Add candidate details
                pdf.cell(0, 8, f"Candidate: {candidate_row['Candidate']}", ln=True)
                pdf.cell(0, 8, f"Final Score: {candidate_row['Score(%)']}%", ln=True)
                pdf.cell(0, 8, f"Semantic Similarity: {candidate_row['Semantic(%)']}%", ln=True)
                pdf.cell(0, 8, f"Keyword Coverage: {candidate_row['Coverage(%)']}%", ln=True)
                pdf.cell(0, 8, f"Experience: {candidate_row['Experience']}", ln=True)
                pdf.cell(0, 8, f"Education: {candidate_row['Education']}", ln=True)
                
                if show_quality_score:
                    pdf.cell(0, 8, f"Resume Quality: {candidate_row['Quality Score']}%", ln=True)
                
                pdf.ln(5)
                pdf.cell(0, 8, "Summary:", ln=True)
                pdf.multi_cell(0, 6, candidate_row['Summary'])
                
                if keyword_toggle:
                    pdf.ln(5)
                    pdf.cell(0, 8, "Matched Keywords:", ln=True)
                    pdf.multi_cell(0, 6, candidate_row['Matched'])
                
                pdf_output = pdf.output(dest='S').encode('latin-1', errors='ignore')
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_output,
                    file_name=f"{selected_candidate}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Resume Ranker Pro+ | Advanced AI-Powered Recruitment Analytics</p>
        <p>Built with ❤️ using Streamlit, scikit-learn, and Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)
