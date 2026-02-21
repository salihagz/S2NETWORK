# model_egitimi.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

print("--- 1. Veri Seti Hazırlanıyor ---")
# Normalde burası veritabanından gelir. Şimdilik öğretiyoruz.
data = {
    'metin': [
        "Python ile görüntü işleme ve derin öğrenme modelleri geliştiriyorum", # Yapay Zeka
        "Yapay zeka algoritmaları ve makine öğrenmesi üzerine çalışıyorum",     # Yapay Zeka
        "Veri madenciliği ve doğal dil işleme nlp projeleri yapıyorum",         # Yapay Zeka
        "React ve Node.js kullanarak web siteleri tasarlıyorum css html",       # Web
        "Frontend geliştirme ve backend api yazılımı yapıyorum",                # Web
        "Arduino ve raspberry pi ile robotik kodlama yapıyorum sensörler",      # Gömülü
        "Gömülü sistemler c++ ve mikrodenetleyici programlama",                 # Gömülü
        "Mobil uygulama geliştirme flutter dart android studio",                # Mobil
        "IOS ve android için native mobil uygulamalar yazıyorum swift kotlin"   # Mobil
    ],
    'kategori': [
        "Yapay Zeka", "Yapay Zeka", "Yapay Zeka",
        "Web Geliştirme", "Web Geliştirme",
        "Gömülü Sistemler", "Gömülü Sistemler",
        "Mobil Uygulama", "Mobil Uygulama"
    ]
}

df = pd.DataFrame(data)

print("--- 2. Metinler Sayılara Çevriliyor (TF-IDF) ---")
# Tezde kullanılan yöntem: TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['metin'])

print("--- 3. SVM Algoritması Eğitiliyor ---")
# Tezde en başarılı bulunan algoritma: SVM
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, df['kategori'])

print("--- 4. Model Kaydediliyor ---")
# Eğitilmiş zekayı dosyaya kaydediyoruz.
joblib.dump(classifier, 'egitilmis_model.pkl')
joblib.dump(vectorizer, 'vektorlestirici.pkl')

print("BAŞARILI: Yapay zeka eğitildi ve dosyalar oluşturuldu!")

