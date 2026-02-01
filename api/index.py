from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import textstat
from textblob import TextBlob
import nltk

# Download resource NLTK (Hanya saat runtime pertama kali)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = FastAPI()

# Definisi format JSON yang diterima
class ArticleRequest(BaseModel):
    title: str
    content: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Quality Gatekeeper (Python/FastAPI)",
        "version": "1.0.0"
    }

@app.post("/api/internal")
def audit_article(item: ArticleRequest):
    text = item.content
    score = 100
    feedback = []
    
    # --- 1. CEK PANJANG KONTEN ---
    word_count = len(text.split())
    if word_count < 300:
        score -= 30
        feedback.append(f"❌ THIN CONTENT: Terlalu pendek ({word_count} kata). Minimal 300+.")
    elif word_count < 600:
        score -= 10
        feedback.append(f"⚠️ LOW DEPTH: Agak pendek ({word_count} kata). Tambah detail.")

    # --- 2. CEK SUBJEKTIVITAS (OPINI) ---
    # Google suka konten yang punya "Perspective" (E-E-A-T)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity # 0.0 (Fakta) - 1.0 (Opini)
    
    if subjectivity < 0.15:
        score -= 15
        feedback.append(f"❌ ROBOTIC: Terlalu kaku/datar (Subjektivitas {subjectivity:.2f}). Tambah opini pribadi 'Menurut saya...'.")
    elif subjectivity > 0.85:
        score -= 10
        feedback.append("⚠️ BIAS: Terlalu curhat/subjektif. Tambah fakta.")

    # --- 3. CEK KETERBACAAN (READABILITY) ---
    # Flesch Reading Ease: Makin tinggi makin mudah dibaca
    try:
        readability = textstat.flesch_reading_ease(text)
        if readability < 30:
            score -= 10
            feedback.append("⚠️ HARD TO READ: Kalimat terlalu rumit/panjang. Pecah kalimat panjang.")
    except:
        pass # Skip jika error bahasa

    # --- KEPUTUSAN FINAL ---
    final_status = score >= 80

    return {
        "status": final_status, # True jika Lolos, False jika butuh revisi
        "score": max(0, score),
        "feedback": feedback,
        "meta": {
            "words": word_count,
            "subjectivity": f"{subjectivity:.2f}",
            "readability": readability
        }
    }
