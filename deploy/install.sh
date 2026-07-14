#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Запустите установщик от имени root." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOME="${OKF_ZVEC_HOME:-/opt/okf-zvec-search}"

apt-get update
apt-get install -y python3 python3-venv ca-certificates openssl

install -d -m 0755 \
  "${APP_HOME}/config" \
  "${APP_HOME}/data/okf" \
  "${APP_HOME}/data/db"

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
echo "Токены сохранены в ${APP_HOME}/config/."

OKF_ZVEC_HOME="${APP_HOME}" "${APP_HOME}/.venv/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer

for name in (
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    model = SentenceTransformer(name)
    print(name, model.get_sentence_embedding_dimension())
PY

install -m 0644 \
  "${PROJECT_DIR}/deploy/okf-zvec-search.service" \
  /etc/systemd/system/okf-zvec-search.service
systemctl daemon-reload
echo "Установка завершена. Запуск: systemctl enable --now okf-zvec-search"
