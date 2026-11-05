# 📄 RAG Belge Asistanı (Full-Stack Document QA & Intelligent Routing System)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_16-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)

**RAG Belge Asistanı**, kullanıcıların yüklediği **PDF, DOCX ve TXT** formatındaki belgeler üzerinde yüksek doğruluklu soru-cevap, adım adım özetleme ve çapraz dil (Cross-Lingual) bilgi çıkarımı yapan uçtan uca (Full-Stack) bir **Retrieval-Augmented Generation (RAG)** sistemidir.

Klasik vektör arama sistemlerindeki "anahtar kelime kaçırma", "farklı dildeki belgeleri eşleştirememe" ve "kaynaksız/halüsinasyonlu yanıt üretme" problemlerini; **Çok Dilli Hibrit Arama (Vektör + Tam Metin Araması + RRF)**, **Akıllı Sorgu Yönlendirme (Query Routing)** ve **Doğrulanabilir Satır İçi Kaynak Gösterimi (Inline Citations)** mimarisiyle çözer.

---

## 🌟 Temel Özellikler

* **🔍 Hibrit Arama Motoru (Vector + Full-Text Search + RRF):**
  * **Anlamsal Arama (Semantic):** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 boyutlu) yerel modeli ve `pgvector` HNSW/kosinüs indeksi ile Türkçe sorgularla İngilizce belgeler arasında bile yüksek anlamsal eşleşme sağlar.
  * **Anahtar Kelime Araması (Lexical):** PostgreSQL `tsvector` ve GIN indeksleri ile özel terimleri, kısaltmaları ve birebir kelimeleri yakalar.
  * **Reciprocal Rank Fusion (RRF):** İki arama kolundan gelen adayları ağırlıklı sıralama füzyonu (`fuse_ranked_chunks`) ve dinamik eşik filtreleriyle birleştirir.
* **🧭 Akıllı Sorgu Yönlendirme (Intelligent Query Router):**
  * Kullanıcının niyetini analiz ederek sorguyu otomatik olarak en uygun rotaya yönlendirir:
    * `🔍 Hibrit arama`: Spesifik bilgi ve kavram sorularında ilgili sayfa/parçaları getirir.
    * `📄 Belge özeti`: Belgenin bütününe dair özet taleplerinde yükleme anında çıkarılan yapılandırılmış belge özetlerini ve konu başlıklarını kullanır.
    * `📑 Tam metin`: Kısa belgelerde bağlam bütünlüğü gerektiren analizleri doğrudan işler.
* **📚 Doğrulanabilir Kaynak Gösterimi (Inline Citations & Source Cards):**
  * Model, ürettiği her bilginin sonuna `[1]`, `[2]` biçiminde satır içi atıf ekler.
  * Yanıtın altındaki açılır **Kaynaklar** panelinde; ilgili parçanın hangi belgeden, kaçıncı sayfadan geldiği, benzerlik skoru (`0.23`, `0.37` vb.), eşleşme türü (`anlamsal + anahtar kelime`) ve orijinal metin kesiti (snippet) şeffaf biçimde sunulur.
* **🛡️ Halüsinasyon Koruması & Prompt Guardrails:**
  * Bağlamda yer almayan konularda bilgi uydurmayı reddeder, belge içi prompt enjeksiyonlarına (Prompt Injection) karşı bağlamı güvenilmeyen veri olarak izole eder.
* **🔐 Çok Kullanıcılı Oturum & Kalıcı Sohbet Hafızası:**
  * JWT tabanlı kimlik doğrulama (`Giriş Yap / Kayıt Ol`), kullanıcı bazlı izole belge kütüphanesi, çoklu sohbet oturumu yönetimi ve canlı yanıt akışı (SSE Streaming) desteklenir.

---

## 🖼️ Ekran Görüntüleri ve Uygulama Akışı

### 1. Kullanıcı Kimlik Doğrulama (JWT Authentication)
Kullanıcılar kendi belge kütüphanelerine ve geçmiş sohbet oturumlarına güvenli şekilde erişmek için sisteme kayıt olur veya giriş yapar. Her kullanıcının verisi veritabanı düzeyinde (`user_id`) izole edilir.

![Kimlik Doğrulama Ekranı](assets/01_auth_screen.png)

---

### 2. Belge Yükleme ve Arama Parametreleri
Sol panel üzerinden **PDF, DOCX ve TXT** (25 MB'a kadar) belgeler yüklenir. Kullanıcı, RAG motorunun getireceği parça sayısını (`top_k`) ve yanıtların canlı akışla (`streaming`) yazılıp yazılmayacağını anlık olarak ayarlayabilir.

| Belge Yükleme & RAG Ayarları | Sohbet Oturumu Yönetimi |
| :---: | :---: |
| ![Sol Panel Belge Yükleme](assets/02_sidebar_upload.png) | ![Sohbetler Sekmesi](assets/03_chat_sessions_empty.png) |

---

### 3. Ana Çalışma Alanı ve Kapsam Filtreleme (Scope Selection)
Kullanıcılar aramayı tüm belgeler üzerinde yapabileceği gibi, üstteki **Sohbet kapsamı** seçicisiyle yalnızca belirli bir belge veya belge grubu içinde daraltabilir. Ayrıca arayüz Sistem, Açık (Light) ve Koyu (Dark) tema modlarını destekler.

![Ana Çalışma Alanı](assets/04_main_workspace.png)

<details>
<summary>🎨 <b>Tema ve Görünüm Ayarları (Görmek için tıklayın)</b></summary>
<br>

![Tema Ayarları](assets/05_theme_settings.png)
</details>

---

### 4. Otomatik Belge İşleme, Parçalama (Chunking) ve Özet Çıkarımı
Yüklenen belgeler arka planda metne dönüştürülür, sayfa numaraları korunarak anlamlı parçalara (chunks) bölünür, vektörleştirilir ve her belge için otomatik olarak **Özet ve Konular** metaverisi oluşturulur. Örneğin 47 sayfalık `Agile Software Engineering.pdf` 72 parçaya, 26 sayfalık `Software Products.pdf` 43 parçaya ayrılarak hazır hale getirilmiştir.

![İşlenmiş Belgeler](assets/06_processed_documents.png)

---

### 5. Çok Dilli Hibrit Arama ve Doğrulanabilir Kaynak Gösterimi (Citations)
Kullanıcı **Türkçe** olarak *"moore vizyonu nedir"* sorusunu sorduğunda, sistem **İngilizce** `1. Software Products.pdf` belgesinin **13. sayfasındaki** *"Moore's vision template"* slaytını `anlamsal + anahtar kelime` hibrit eşleşmesiyle yakalar. Yanıtı maddeler halinde Türkçe sentezler, her maddeye `[1]` atfını ekler ve alt panelde tüm kaynak detaylarını (`skor`, `sayfa`, `snippet`) listeler.

![Hibrit Arama ve Kaynaklar](assets/07_hybrid_search_citations.png)

---

### 6. Akıllı Sorgu Yönlendirme: Belge Özeti Rotası ve Kalıcı Sohbet Geçmişi
Kullanıcı *"dosyayı adım adım özetle"* gibi genel bir istekte bulunduğunda, Sorgu Yönlendirici (Router) otomatik olarak **`Rota: 📄 Belge özeti`** moduna geçer ve belgenin tüm ana başlıklarını adım adım özetler. Oluşturulan tüm diyaloglar sol menüdeki **Sohbetler** sekmesinde otomatik başlıkla kaydedilir.

| Otomatik Kaydedilen Sohbet Geçmişi | Akıllı Rota: Belge Özeti (`📄 Belge özeti`) |
| :---: | :---: |
| ![Aktif Sohbet Geçmişi](assets/08_chat_history_active.png) | ![Belge Özeti Rotası](assets/09_summary_routing.png) |

---

## 🏗️ Sistem Mimarisi

```mermaid
flowchart TD
    User([Kullanici]) -->|JWT Auth & HTTP| API[FastAPI Backend]
    
    subgraph Ingestion [Belge Isleme Hatti]
        API -->|PDF / DOCX / TXT| Parser[Metin ve Sayfa Ayristirici]
        Parser --> Chunker[Recursive Text Chunker]
        Chunker --> Embedder[Yerel Embedding Modeli]
        Chunker --> DocSummary[LLM Belge Ozeti]
        Embedder -->|Vektor + tsvector| DB[(PostgreSQL + pgvector)]
        DocSummary -->|Metadata| DB
    end

    subgraph Retrieval [Hibrit Arama Motoru]
        API --> Router{Query Router}
        Router -->|Detay Sorusu| Hybrid[Hibrit Arama Servisi]
        Router -->|Ozet Talebi| SummaryRoute[Belge Ozeti Rotasi]
        
        Hybrid -->|Kosinus Benzerligi| VecSearch[Vektor Aramasi]
        Hybrid -->|FTS / GIN| FTSSearch[Anahtar Kelime Aramasi]
        VecSearch & FTSSearch --> RRF[RRF ve Esik Filtresi]
        RRF --> Context[Atifli Baglam]
    end

    Context & SummaryRoute --> LLM[Google Gemini API]
    LLM -->|Atifli Yanit ve Kaynaklar| User

---

## 🚀 Kurulum ve Çalıştırma

Projeyi **Docker Compose** ile tek komutla tüm bağımlılıklarıyla (PostgreSQL `pgvector` + FastAPI + Streamlit) ayağa kaldırabilirsiniz.

### 1. Repoyu Klonlayın ve Ortam Değişkenlerini Hazırlayın
```bash
git clone https://github.com/akdenizenes/rag-document-assistant.git
cd rag-document-assistant
cp .env.example .env
```

`.env` dosyasını açarak kendi `GEMINI_API_KEY` anahtarınızı girin:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RETRIEVAL_MIN_SCORE=0.15
HYBRID_VECTOR_WEIGHT=0.85
HYBRID_FTS_WEIGHT=0.15
```

### 2. Docker ile Başlatın
```bash
docker compose up -d --build
```
* **Streamlit Arayüzü:** `http://localhost:8501`
* **FastAPI Swagger Dokümantasyonu:** `http://localhost:8000/docs`
* **Sağlık Kontrolü (Healthcheck):** `http://localhost:8000/health/ready`

---

## 📂 Proje Dizin Yapısı

```text
rag-document-assistant/
├── app/
│   ├── main.py                 # FastAPI uygulama girişi ve yaşam döngüsü (lifespan)
│   ├── config.py               # Ortam değişkenleri ve RAG hiperparametreleri
│   ├── database.py             # Async SQLAlchemy & PostgreSQL bağlantı yönetimi
│   ├── models/                 # Veritabanı tabloları (User, Document, Chunk, ChatSession)
│   ├── routers/                # API uç noktaları (auth, documents, chat, health)
│   ├── schemas/                # Pydantic veri doğrulama ve yanıt şemaları
│   └── services/
│       ├── embedding_service.py # Çok dilli yerel SentenceTransformer servisi
│       ├── retrieval_service.py # Vektör + FTS hibrit arama ve RRF füzyon algoritması
│       ├── query_router.py      # Akıllı sorgu sınıflandırma ve yönlendirme
│       ├── citations.py         # Atıf ayıklama ve kaynak referans eşleştirme
│       ├── prompts.py           # Sistem kuralları ve numaralı bağlam oluşturucu
│       └── llm_service.py       # Google GenAI (Gemini) üretim ve akış (streaming) servisi
├── frontend/
│   └── streamlit_app.py        # Hataya dayanıklı, çoklu oturum destekli Streamlit arayüzü
├── scripts/                    # Veritabanı şema migrasyon betikleri (v1, v2, v3)
├── tests/                      # Birim ve entegrasyon testleri (pytest)
├── assets/                     # Uygulama ekran görüntüleri
├── docker-compose.yml          # Çoklu konteyner orkestrasyonu (db, api, frontend)
└── requirements.txt            # Python kütüphane bağımlılıkları
```

---

## 🧪 Testleri Çalıştırma

```bash
pytest -q
```