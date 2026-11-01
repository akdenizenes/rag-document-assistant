# Adım 4 — Özet (yapılanlar / yapılmayanlar)

## Yapıldı
- 4.1 Sohbet geçmişi: `ChatSession`/`ChatMessage` modelleri (tablolar `init_db`/`create_all` ile otomatik oluşur, migrasyon yok),
  `/sessions` (POST/GET/GET{id}/PATCH/DELETE), `/chat` ve `/chat/stream` için `session_id`, Streamlit "Sohbetler" sekmesi.
- 4.2 Çoklu belge kapsamı: `document_ids` (backend + `st.multiselect`). Tam metin rotası en fazla `FULL_TEXT_MAX_DOCS` (3) belge.
- 4.3 Reranker: `RERANK_MODE=off|score|cross_encoder` (`app/services/rerank_service.py`). Varsayılan `score` (modelsiz).
- 4.4 Rate limit (bağımlılıksız, süreç içi, kullanıcı başına), merkezi hata yakalayıcılar + `X-Request-ID`, `/health/ready`,
  Streamlit bölüm bazlı hata yakalama ve API istemcisinde GET yeniden deneme.

## YAPILMADI / DOĞRULANMADI
- `slowapi` kullanılmadı (pypi bu ortamda engelliydi). Sayaç süreç içi: `--workers N` ile sınır N katına çıkar; Redis yok.
- Cross-encoder modeli hiç indirilip çalıştırılmadı (yalnızca sahte skorlayıcıyla test edildi). Model adı doğrulanmadı.
- SQLAlchemy/FastAPI/Streamlit kodu BU ORTAMDA ÇALIŞTIRILMADI (kurulamadı): yeni modeller, `/sessions`, `chat.py`,
  `errors.py`, `rate_limit.py`, `streamlit_app.py` yalnızca derleme + ruff ile kontrol edildi. İlk çalıştırmada hata çıkabilir.
- Test edilenler: router, RRF, rerank mantığı, rate limiter, prompt/özet ayrıştırma, API istemcisi, rota geri dönüşleri
  (sahte DB ile). Gerçek PostgreSQL'de yalnızca Adım 3 SQL'i denendi; pgvector hiç denenmedi.
- Kullanıcı kimliği hâlâ tek "demo" kullanıcı (JWT yok): sohbet geçmişi ve rate limit herkes için ortak.
- Oturum içi mesaj düzenleme/yeniden üretme, sohbet başlığı yeniden adlandırma arayüzü (API'de PATCH var) yok.
- Eski testler `document_id` → `document_ids` için güncellendi; `scripts/` ve DB için entegrasyon testi yok.
