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

# --- SETUP RESOURCE (Tetap Ringan) ---
NLTK_RESOURCES = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']
for resource in NLTK_RESOURCES:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource.startswith('punkt') else f'corpora/{resource}' if resource == 'stopwords' else f'help/{resource}' if resource == 'tagsets' else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

app = FastAPI()

class ArticleRequest(BaseModel):
    title: str = ""
    content: str
    language: str = "english"

# --- FUNGSI MATEMATIKA CANGGIH (BARU) ---

def calculate_entropy(text, n=2):
    """
    Menghitung SHANNON ENTROPY dari N-Grams.
    Ini adalah proxy ringan untuk 'Perplexity'.
    
    - Entropy Rendah (< 3.5) = Teks sangat terprediksi (Ciri khas AI/Robot).
    - Entropy Tinggi (> 4.5) = Teks acak/variatif (Ciri khas Manusia).
    """
    words = word_tokenize(text.lower())
    if len(words) < 2: return 0
    
    # Buat N-grams (pasangan kata)
    n_grams = list(ngrams(words, n))
    # Hitung frekuensi
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
        # NNP = Proper Noun (Nama Spesifik), CD = Cardinal Number (Angka)
        # Manusia suka fakta (Angka & Nama). AI suka deskripsi umum.
        entities = [w for w, pos in pos_tags if pos in ['NNP', 'NNPS', 'CD']]
        return len(entities) / len(words)
    except:
        return 0.1

# --- ENDPOINT 1: QUALITY CHECK (THE MATHEMATICAL JUDGE) ---
@app.post("/api/internal")
def quality_check(item: ArticleRequest):
    text = item.content
    score = 100
    feedback = []
    
    # Pre-processing
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    clean_words = [w for w in words if w.isalnum()]
    word_count = len(clean_words)
    
    # --- 1. ENTROPY CHECK (Detektor AI Paling Akurat di Python Ringan) ---
    # Menggunakan Trigram Entropy (3 kata berurutan)
    entropy_score = calculate_entropy(text, n=3)
    
    # Ambang batas riset: Teks AI biasanya di bawah 8.0 (untuk trigram text panjang)
    # Teks Manusia biasanya di atas 9.0
    # *Angka ini relatif tergantung panjang teks, kita pakai rasio aman*
    
    if entropy_score < 7.5: 
        score -= 25
        feedback.append(f"❌ AI PATTERN DETECTED: Entropi bahasa terlalu rendah ({entropy_score:.2f}). Pola kalimat terlalu mudah ditebak. Gunakan struktur bahasa yang lebih 'kacau' atau idiom.")
    elif entropy_score < 8.5:
        score -= 10
        feedback.append(f"⚠️ TOO SMOOTH: Tulisan terlalu mulus ({entropy_score:.2f}). Tambahkan kejutan dalam kalimat.")

    # --- 2. ENTITY DENSITY (Fakta) ---
    ent_density = get_entity_density(clean_words)
    if ent_density < 0.09: # 9% kata harus berupa Nama/Angka
        score -= 20
        feedback.append(f"❌ FLUFF: Minim Fakta/Data (Density {ent_density*100:.1f}%). Tambahkan Angka, Harga, Spesifikasi, atau Nama Lokasi.")

    # --- 3. BASIC CHECKS (Tetap Dipertahankan) ---
    if word_count < 450:
        score -= 30
        feedback.append(f"❌ TOO SHORT: ({word_count} kata). Google News butuh kedalaman.")
    
    # TTR (Repetisi)
    unique_words = set([w.lower() for w in clean_words])
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    if ttr < 0.39:
        score -= 15
        feedback.append(f"❌ REPETITIVE: Kosakata itu-itu saja (TTR {ttr:.2f}).")

    # Subjectivity
    blob = TextBlob(text)
    subj = blob.sentiment.subjectivity
    if subj < 0.15:
        score -= 10
        feedback.append("⚠️ ROBOT TONE: Terlalu objektif. Masukkan opini pribadi.")

    # --- FINAL VERDICT ---
    return {
        "status": score >= 85,
        "score": max(0, score),
        "feedback": feedback,
        "advanced_metrics": {
            "entropy_score": f"{entropy_score:.2f} (Target > 9.0)",
            "entity_density": f"{ent_density*100:.1f}% (Target > 9%)",
            "ttr": f"{ttr:.2f}"
        }
    }

# --- ENDPOINT 2: PLAGIARISM (SMART FILTER) ---
@app.post("/api/plagiat")
def plagiarism_check(item: ArticleRequest):
    text = item.content
    sentences = sent_tokenize(text)
    
    # Strategi 2026: Cari 'Fingerprint' Unik
    # Cari kalimat yang punya angka spesifik (misal: "baterai 5000mAh", "naik 20%")
    # Plagiarisme angka lebih fatal daripada plagiarisme kata sifat.
    
    candidates = []
    for s in sentences:
        words = s.split()
        if len(words) > 8:
            weight = 0
            if any(char.isdigit() for char in s): weight += 5 # Prioritas Angka
            if any(w[0].isupper() for w in words[1:]): weight += 3 # Prioritas Nama
            candidates.append((weight, s))
            
    # Sort ambil yang paling 'kaya data'
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c[1] for c in candidates[:3]] # Cek 3 teratas
    
    matches = []
    if not top_candidates:
        return {"status": True, "message": "Teks aman (terlalu umum).", "matches": []}

    with DDGS() as ddgs:
        for sentence in top_candidates:
            try:
                # Search dengan operator fuzzy (tanpa tanda kutip) agar lebih sensitif terhadap rewrite
                # Lalu kita cek kemiripan manual
                query = sentence[:100]
                results = list(ddgs.text(query, max_results=1))
                if results:
                    # Cek sederhana apakah snippet hasil mirip
                    snippet = results[0]['body']
                    # Jika snippet mengandung setidaknya 50% kata unik dari kalimat kita -> Plagiat
                    unique_sen = set(sentence.lower().split())
                    unique_snip = set(snippet.lower().split())
                    overlap = len(unique_sen.intersection(unique_snip)) / len(unique_sen)
                    
                    if overlap > 0.6: # Ambang batas kemiripan 60%
                        matches.append({
                            "sentence": sentence[:50] + "...",
                            "source": results[0]['href'],
                            "similarity": f"{overlap*100:.0f}%"
                        })
            except:
                continue

    return {
        "status": len(matches) == 0,
        "matches": matches
    }