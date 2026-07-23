# ⚡ Yankı AI Ultra

Akıllı, hızlı ve çok yönlü Türkçe yapay zeka asistanı.

## 🚀 Render'da Deploy Etme Adımları

1. GitHub'da yeni repo oluştur, bu dosyaları yükle
2. [render.com](https://render.com) git, "New Web Service" de
3. GitHub reposunu bağla
4. Ortam değişkenlerini ekle (Environment Variables):
   - `SECRET_KEY` = rastgele uzun bir string
   - `GROQ_API_KEY` = Groq'dan alacağın ücretsiz API key
5. Deploy et

## 🔑 API Key Alma (ÜCRETSİZ)

**Groq (Önerilen - çok hızlı):**
1. [console.groq.com](https://console.groq.com) git
2. Üye ol (ücretsiz)
3. API Keys > Create API Key
4. Key'i Render Environment Variables'a ekle

**Gemini (Opsiyonel):**
1. [aistudio.google.com](https://aistudio.google.com) git
2. API Key oluştur
3. Render'a `GEMINI_API_KEY` olarak ekle

## ⚠️ Önemli Notlar

- **Ollama** sadece local bilgisayarında çalışır, Render'da çalışmaz.
- Render'da SQLite kullanıyorsan, ücretsiz planda veritabanı her deploy'da sıfırlanır. Kalıcı veri için Render PostgreSQL ekle.
- Kod çalıştırma, web scraping, görsel analiz ve oyun test **API key'siz** çalışır.

## 👑 Kurucu

`ilarslanelif8@gmail.com` ile giriş yapan kullanıcı "Kurucu Modu" ile özel yetkilere sahip olur.
