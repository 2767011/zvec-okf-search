#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Запустите установщик от имени root." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOME="${OKF_ZVEC_HOME:-/opt/okf-zvec-search}"
SERVICE_USER="okf-zvec"

apt-get update
apt-get install -y python3 python3-venv ca-certificates openssl

if ! getent passwd "${SERVICE_USER}" >/dev/null; then
  useradd --system --home-dir "${APP_HOME}" --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

install -d -m 0755 \
  "${APP_HOME}/config" \
  "${APP_HOME}/data/okf" \
  "${APP_HOME}/data/db" \
  "${APP_HOME}/data/huggingface"

python3 -m venv "${APP_HOME}/.venv"
"${APP_HOME}/.venv/bin/pip" install --upgrade pip
"${APP_HOME}/.venv/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch
"${APP_HOME}/.venv/bin/pip" install "${PROJECT_DIR}"

if [[ ! -e /etc/okf-zvec-search.env ]]; then
  install -m 0640 "${PROJECT_DIR}/.env.example" /etc/okf-zvec-search.env
fi

if [[ ! -s "${APP_HOME}/config/service-token" ]]; then
  openssl rand -hex 32 > "${APP_HOME}/config/service-token"
  chmod 0600 "${APP_HOME}/config/service-token"
fi
if [[ ! -s "${APP_HOME}/config/search-token" ]]; then
  openssl rand -hex 32 > "${APP_HOME}/config/search-token"
  chmod 0600 "${APP_HOME}/config/search-token"
fi
if [[ ! -s "${APP_HOME}/config/admin-token" ]]; then
  openssl rand -hex 32 > "${APP_HOME}/config/admin-token"
fi
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_HOME}/config" "${APP_HOME}/data"
chmod 0600 \
  "${APP_HOME}/config/service-token" \
  "${APP_HOME}/config/search-token" \
  "${APP_HOME}/config/admin-token"
echo "Токены сохранены в ${APP_HOME}/config/."

INDEX_MODELS="${OKF_ZVEC_INDEX_MODELS:-e5}"
runuser -u "${SERVICE_USER}" -- env \
  HOME="${APP_HOME}" \
  HF_HOME="${APP_HOME}/data/huggingface" \
  OKF_ZVEC_HOME="${APP_HOME}" \
  OKF_ZVEC_INDEX_MODELS="${INDEX_MODELS}" \
  "${APP_HOME}/.venv/bin/python" - <<'PY'
import os
from sentence_transformers import SentenceTransformer

models = {
    "e5": "intfloat/multilingual-e5-small",
    "paraphrase": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
keys = models if os.environ["OKF_ZVEC_INDEX_MODELS"].casefold() == "all" else (
    key.strip() for key in os.environ["OKF_ZVEC_INDEX_MODELS"].split(",")
)
for key in keys:
    name = models[key]
    model = SentenceTransformer(name)
    print(name, model.get_sentence_embedding_dimension())
PY

install -m 0644 \
  "${PROJECT_DIR}/deploy/okf-zvec-search.service" \
  /etc/systemd/system/okf-zvec-search.service
systemctl daemon-reload
echo "Установка завершена. Запуск: systemctl enable --now okf-zvec-search"
