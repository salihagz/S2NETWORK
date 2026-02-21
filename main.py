# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yake

# 1. Eğitilmiş Modelleri Yükle
print("Sistem başlatılıyor...")
try:
    model = joblib.load('egitilmis_model.pkl')
    vectorizer = joblib.load('vektorlestirici.pkl')
    print("Modeller başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Modeller yüklenemedi. Lütfen önce model_egitimi.py dosyasını çalıştırın. Hata: {e}")

app = FastAPI()

# Gelen veri formatı
class VeriModeli(BaseModel):
    metin: str

@app.post("/analiz")
def analiz_et(veri: VeriModeli):
    gelen_metin = veri.metin.lower()
    
    # A) Kategori Tahmini (SVM)
    vektor = vectorizer.transform([gelen_metin])
    tahmin_kategori = model.predict(vektor)[0]
    
    # B) Yetkinlik Çıkarımı (YAKE)
    kw_extractor = yake.KeywordExtractor(lan="tr", n=1, dedupLim=0.9, top=5)
    anahtar_kelimeler = kw_extractor.extract_keywords(gelen_metin)
    bulunan_ozellikler = [kelime[0] for kelime in anahtar_kelimeler]
    
    # C) Basit Öneri Sistemi
    oneriler = []
    if tahmin_kategori == "Yapay Zeka":
        oneriler = ["tensorflow", "keras", "python", "istatistik"]
    elif tahmin_kategori == "Web Geliştirme":
        oneriler = ["docker", "sql", "react", "security"]
    elif tahmin_kategori == "Gömülü Sistemler":
        oneriler = ["pcb", "altium", "rtos", "c"]
    elif tahmin_kategori == "Mobil Uygulama":
        oneriler = ["firebase", "ui/ux", "swift", "kotlin"]
    
    return {
        "kategori": tahmin_kategori,
        "tespit_edilen": bulunan_ozellikler,
        "onerilen_eksik_beceriler": oneriler
    }