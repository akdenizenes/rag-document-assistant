# Docker ile Production Kurulumu

> Volume adları `<klasör-adı>_pgdata` biçimindedir; aşağıdaki `rag-assistant_` önekini proje klasörünün adıyla değiştir (`docker volume ls`).

Servisler: `db` (pgvector/pgvector:pg16) → `api` (FastAPI + Tesseract/Poppler) → `frontend` (Streamlit),
hepsi `rag_net` ağında. Veri üç named volume'da: `pgdata`, `uploads_data`, `hf_cache`.

## Gereksinimler
Docker Engine 24+ ve Docker Compose v2. Önerilen: 4 GB+ RAM (embedding modeli + OCR), 6 GB+ disk.

## Hızlı başlangıç
```bash
# 1) Sırları doldur
nano .env.docker            # POSTGRES_PASSWORD, JWT_SECRET_KEY, LLM_PROVIDER/LLM_API_KEY
openssl rand -hex 32        # JWT_SECRET_KEY için

# 2) Tek komut
docker compose up --build -d

# 3) İzle (ilk açılış: model + migrasyon, 1-2 dk sürebilir)
docker compose logs -f api
docker compose ps           # api "healthy" olunca frontend başlar
```
Arayüz: http://localhost:8501 · Swagger: http://127.0.0.1:8000/docs (yalnızca sunucunun kendisinden)

`.env.docker` içindeki `FRONTEND_PORT`/`API_PORT` gibi compose-düzeyi değişkenleri uygulamak için:
`docker compose --env-file .env.docker up --build -d`

## API açılış akışı (`entrypoint.sh`)
1. Güvenlik kontrolü: `JWT_SECRET_KEY`/`POSTGRES_PASSWORD` örnek (`CHANGE_ME`) veya kısa ise API **açılmaz**.
2. `DATABASE_URL`, `POSTGRES_*` değişkenlerinden parolayı URL-encode ederek üretilir.
3. `scripts/docker_db.py wait`: PostgreSQL bağlantı kabul edene kadar bekler (`WAIT_FOR_DB_TIMEOUT`, 90 sn).
4. `docker_db.py bootstrap`: `CREATE EXTENSION vector` + eksik tabloları oluşturur.
5. `migrate_v2.py` → `migrate_v3.py` (ikisi de idempotent; boş veritabanında "değişiklik yok" olur).
6. `uvicorn` (proxy başlıkları, graceful shutdown, `UVICORN_WORKERS`).

Migrasyonu atlamak: `RUN_MIGRATIONS=false`. Başka bir komut çalıştırmak (entrypoint "serve" dışını doğrudan çalıştırır):
```bash
docker compose run --rm api python scripts/backfill_metadata.py --dry-run
docker compose run --rm api python scripts/backfill_metadata.py
```

## Günlük işler
```bash
docker compose logs -f api frontend      # loglar
docker compose restart api               # yeniden başlat
docker compose up --build -d             # kod güncellemesi sonrası
docker compose down                      # durdur (veri KALIR)
docker compose down -v                   # durdur + TÜM VERİYİ SİL (db, yüklenen dosyalar, model önbelleği)

# Yedek / geri yükleme
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' > backup.sql
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" "$POSTGRES_DB"' < backup.sql
docker run --rm -v rag-assistant_uploads_data:/data -v "$PWD":/b alpine tar czf /b/uploads.tgz -C /data .
```

## Yerel LLM (Ollama)
`docker compose --profile llm up --build -d`, ardından `docker compose exec ollama ollama pull qwen2.5:7b`
ve `.env.docker`: `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:7b`, `LLM_BASE_URL=http://ollama:11434`.
Ana makinedeki Ollama için `LLM_BASE_URL=http://host.docker.internal:11434`.

## Uzak sunucu / HTTPS
Yalnızca frontend (8501) dışarı açıktır; API ve DB açılmaz. Önüne TLS sonlandıran bir proxy koy
(Streamlit websocket kullanır; Caddy/Nginx'te websocket'e izin ver). Caddy örneği:
```
rag.ornek.com {
    reverse_proxy localhost:8501
}
```
Güvenlik duvarında yalnızca 80/443 (ve SSH) açık kalsın; 8501'i proxy kullanıyorsan `FRONTEND_PORT` yerine
compose'ta `"127.0.0.1:8501:8501"` yap.

## Dikkat edilecekler
- **Eski yerel kurulumdan geçiş:** önceki compose `ankane/pgvector` (PostgreSQL 15) kullanıyordu.
  Aynı `pgdata` volume'u PostgreSQL 16 ile **açılmaz** ("database files are incompatible"). Eski veriyi
  korumak için eski kurulumdayken `pg_dump` al, `docker volume rm rag-assistant_pgdata`, yeni kurulumu başlat, dump'ı yükle.
  Yerel `./uploads/` klasöründeki dosyalar da `uploads_data` volume'una elle kopyalanmalı.
- **Çoklu worker:** `UVICORN_WORKERS>1` iken rate limiter worker başına ayrı sayar, embedding modeli
  her worker'da ayrı yüklenir ve açılıştaki `recover_stale_documents()` diğer worker'ın işlediği belgeyi "failed" yapabilir.
  Ölçeklemeden önce 1'de bırak.
- **Parolalar:** `.env.docker` içinde `$` kullanma (compose değişken sanır). `openssl rand -hex 24` güvenlidir.
- `.env.docker` `.gitignore`'a eklendi; sunucuya git yerine `scp` ile ya da orada oluşturarak taşı.
- Windows'ta `entrypoint.sh` CRLF olursa imaj derlenirken otomatik düzeltilir (ayrıca `.gitattributes` LF zorlar).
- GPU yok varsayımıyla CPU torch kurulur (imaj ≈ 1.5 GB daha küçük). GPU için
  `--build-arg TORCH_INDEX_URL=https://pypi.org/simple`.
- Hızlı/ağsız derleme: `docker compose build --build-arg PRELOAD_EMBEDDING_MODEL=false api`
  (model ilk çalıştırmada indirilir).

## "Beni hatırla" (kalıcı oturum)
Giriş yapınca JWT, tarayıcıda `rag_auth_token` çerezine yazılır (süre = min(`AUTH_COOKIE_DAYS`=7 gün, token ömrü)).
Sayfa yenilenince frontend çerezdeki token'ı API'nin `GET /auth/me` ucuyla doğrular. Token ömrü
`ACCESS_TOKEN_EXPIRE_MINUTES` ile belirlenir (`.env.docker`'da 10080 = 7 gün). Bu özelliği alırken hem
`api` (yeni `/auth/me` ucu) hem `frontend` (yeni paket) yeniden derlenmelidir:
`docker compose up --build -d`
