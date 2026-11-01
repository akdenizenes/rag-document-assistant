#!/usr/bin/env bash
set -euo pipefail

log() { echo "[entrypoint] $*"; }

if [[ "${1:-}" != "serve" ]]; then
    exec "$@"
fi

# --- 1) Güvenlik: şablondaki varsayılan sırlarla başlatmayı engelle
if [[ "${ALLOW_INSECURE_DEFAULTS:-false}" != "true" ]]; then
    jwt="${JWT_SECRET_KEY:-}"
    pg="${POSTGRES_PASSWORD:-}"
    if [[ -z "$jwt" || "${jwt,,}" == *change_me* || "${jwt,,}" == super-secret-rag-jwt* || ${#jwt} -lt 32 ]]; then
        log "HATA: JWT_SECRET_KEY guvenli degil (bos/ornek deger/32 karakterden kisa)."
        exit 1
    fi
    if [[ -z "$pg" || "${pg,,}" == *change_me* ]]; then
        log "HATA: POSTGRES_PASSWORD ayarlanmamis veya ornek degerde."
        exit 1
    fi
fi

# --- 2) DATABASE_URL oluştur
DATABASE_URL="$(python scripts/docker_db.py url)"
export DATABASE_URL

# --- 3) Veritabanını bekle
log "Veritabani bekleniyor (${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432})..."
python scripts/docker_db.py wait

# --- 4-5) Veritabanı tabloları ve migrasyonlar
if [[ "${RUN_MIGRATIONS:-true}" == "true" ]]; then
    log "Tablolar hazirlaniyor..."
    python scripts/docker_db.py bootstrap
    log "migrate_v2 calisiyor..."
    python scripts/migrate_v2.py
    log "migrate_v3 calisiyor..."
    python scripts/migrate_v3.py
else
    log "RUN_MIGRATIONS=false: migrasyonlar atlandi."
fi

# --- 6) Uvicorn başlat
log "Uvicorn baslatiliyor (workers=${UVICORN_WORKERS:-1})..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${UVICORN_WORKERS:-1}" \
    --proxy-headers \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" \
    --timeout-keep-alive 65 \
    --timeout-graceful-shutdown 25 \
    --log-level "${UVICORN_LOG_LEVEL:-info}"