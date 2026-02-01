from fastapi import FastAPI
from pydantic import BaseModel
import textstat
from textblob import TextBlob
import nltk
import numpy as np
from duckduckgo_search import DDGS
import random
import math
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import os

# ==========================================
# 🔧 PERBAIKAN KHUSUS VERCEL (WAJIB ADA)
# ==========================================

# 1. Tentukan folder download ke /tmp (Satu-satunya folder writable di Vercel)
NLTK_DATA_PATH = "/tmp/nltk_data"

# 2. Buat folder jika belum ada
if not os.path.exists(NLTK_DATA_PATH):
    try:
        os.makedirs(NLTK_DATA_PATH)
    except OSError:
        pass # Abaikan jika sudah ada (race condition)

# 3. Tambahkan /tmp ke jalur pencarian NLTK agar data bisa ditemukan setelah didownload
nltk.data.path.append(NLTK_DATA_PATH)

# 4. Daftar Resource yang Wajib Didownload
NLTK_RESOURCES = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']

print(f"🔄 Checking NLTK resources in {NLTK_DATA_PATH}...")

for resource in NLTK_RESOURCES:
    try:
        # Cek apakah resource sudah ada (biar ga download ulang tiap request)
        # Perhatikan: kita cek folder tokenizers/corpora/taggers sesuai jenis resource
        if resource == 'stopwords':
            nltk.data.find(f'corpora/{resource}')
        elif resource == 'averaged_perceptron_tagger':
            nltk.data.find(f'taggers/{resource}')
        else:
            nltk.data.find(f'tokenizers/{resource}')
            
    except LookupError:
        print(f"⬇️ Downloading {resource}...")
        try:
            # FORCE download ke folder /tmp
            nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
        except Exception as e:
            print(f"⚠️ Failed to download {resource}: {e}")

# ==========================================
# LOGIC ANALISIS (SPAMBRAIN V2)
# ==========================================

app = FastAPI()

class ArticleRequest(BaseModel):
    title: str = ""
    content: str
    language: str = "english"

def calculate_entropy(text, n=2):
    """Menghitung SHANNON ENTROPY (Deteksi Pola AI)"""
    # Gunakan try-except agar tidak crash jika NLTK gagal load saat runtime
    try:
        words = word_tokenize(text.lower())
    except LookupError:
        return 9.0 # Default aman jika NLTK error
        
    if len(words) < 2: return 0
    
    n_grams = list(ngrams(words, n))
    counts = Counter(n_grams)
    total_count = sum(counts.values())
    
    entropy = 0
    for count in counts.values():
        p = count / total_count
        entropy -= p * math.log2(p)
    
    return entropy

def get_entity_density(words):
    try:
        pos_tags = nltk.pos_tag(words)
        entities = [w for w, pos in pos_tags if pos in ['NNP', 'NNPS', 'CD']]
        return len(entities) / len(words)
    except:
        return 0.1

@app.get("/")
def home():
    return {"status": "SpamBrain V2 Online (Fixed NLTK Path)"}

@app.post("/api/internal")
def quality_check(item: ArticleRequest):
    text = item.content
    score = 100
    feedback = []
    
    # Pre-processing aman
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
    except LookupError:
        # Fallback manual jika tokenizer macet total
        sentences = text.split('.')
        words = text.split()

    clean_words = [w for w in words if w.isalnum()]
    word_count = len(clean_words)
    
    # --- 1. ENTROPY CHECK ---
    entropy_score = calculate_entropy(text, n=3)
    if entropy_score < 7.5: 
        score -= 25
        feedback.append(f"❌ AI PATTERN: Entropi terlalu rendah ({entropy_score:.2f}).")
    elif entropy_score < 8.5:
        score -= 10
        feedback.append(f"⚠️ TOO SMOOTH: Tulisan terlalu mulus ({entropy_score:.2f}).")

    # --- 2. ENTITY DENSITY ---
    ent_density = get_entity_density(clean_words)
    if ent_density < 0.09:
        score -= 20
        feedback.append(f"❌ FLUFF: Minim Fakta/Data (Density {ent_density*100:.1f}%).")

    # --- 3. BASIC CHECKS ---
    if word_count < 450:
        score -= 30
        feedback.append(f"❌ TOO SHORT: ({word_count} kata).")
    
    unique_words = set([w.lower() for w in clean_words])
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    if ttr < 0.39:
        score -= 15
        feedback.append(f"❌ REPETITIVE: TTR Rendah ({ttr:.2f}).")

    blob = TextBlob(text)
    subj = blob.sentiment.subjectivity
    if subj < 0.15:
        score -= 10
        feedback.append("⚠️ ROBOT TONE: Terlalu objektif.")

    return {
        "status": score >= 85,
        "score": max(0, score),
        "feedback": feedback,
        "advanced_metrics": {
            "entropy": f"{entropy_score:.2f}",
            "entity_density": f"{ent_density*100:.1f}%",
            "ttr": f"{ttr:.2f}"
        }
    }

@app.post("/api/plagiat")
def plagiarism_check(item: ArticleRequest):
    text = item.content
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')

    # Prioritaskan kalimat dengan Angka/Kapital
    candidates = []
    for s in sentences:
        words = s.split()
        if len(words) > 8:
            weight = 0
            if any(char.isdigit() for char in s): weight += 5
            if any(w[0].isupper() for w in words[1:]): weight += 3
            candidates.append((weight, s))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c[1] for c in candidates[:3]]
    
    matches = []
    if not top_candidates:
        return {"status": True, "message": "Teks aman.", "matches": []}

    with DDGS() as ddgs:
        for sentence in top_candidates:
            try:
                query = sentence[:100]
                results = list(ddgs.text(query, max_results=1))
                if results:
                    snippet = results[0]['body']
                    unique_sen = set(sentence.lower().split())
                    unique_snip = set(snippet.lower().split())
                    overlap = len(unique_sen.intersection(unique_snip)) / len(unique_sen)
                    
                    if overlap > 0.6:
                        matches.append({
                            "sentence": sentence[:50] + "...",
                            "source": results[0]['href'],
                            "similarity": f"{overlap*100:.0f}%"
                        })
            except:
                continue

    return {"status": len(matches) == 0, "matches": matches}