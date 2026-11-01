# Çoklu Format (.pdf, .docx, .txt) ve OCR Desteği

## 1) Python paketleri
    pip install -r requirements.txt     # python-docx, pytesseract, pdf2image, Pillow eklendi

## 2) Sistem bağımlılıkları (yalnızca taranmış PDF OCR'ı için)
Ubuntu/Debian:
    sudo apt update
    sudo apt install -y tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng poppler-utils

macOS (Homebrew):
    brew install tesseract tesseract-lang poppler

Windows:
  - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki  (kurulumda "Turkish" dilini seç, PATH'e ekle)
  - Poppler:   https://github.com/oschwartz10612/poppler-windows/releases  (bin klasörünü PATH'e ekle)

Docker imajı kullanıyorsan Dockerfile'a ekle:
    RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng poppler-utils \
        && rm -rf /var/lib/apt/lists/*

Kontrol:
    tesseract --version && tesseract --list-langs   # tur ve eng görünmeli
    pdftoppm -v

OCR kurulu değilse uygulama ÇÖKMEZ: metin katmanlı PDF/DOCX/TXT normal çalışır; yalnızca
taranmış sayfalar için belge "failed" olur ve nedeni (ör. "Tesseract kurulu değil") gösterilir.

## 3) Veritabanı
Şema değişikliği YOKTUR (content_type zaten String(100); yeni MIME türleri sığar).

## 4) Mevcut yüklemeler
Eski belgeler `uploads/<id>.pdf` olarak kalır ve aynen çalışır. Yeni dosyalar `<id>.pdf|docx|txt`.

## 5) Davranış notları
- OCR sayfa bazlıdır: yalnızca metni < OCR_MIN_CHARS_PER_PAGE (25) karakter olan sayfalara uygulanır.
- DOCX/TXT'de gerçek sayfa yoktur; metin ~3000 karakterlik "bölümlere" ayrılır, arayüz "bölüm N" gösterir.
- TXT çözme sırası: UTF-8 (BOM'lu/BOM'suz) -> UTF-16 (BOM varsa) -> cp1254 -> ISO-8859-9.
- Eski .doc desteklenmez (yalnızca .docx).
