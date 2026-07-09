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
  "${APP_HOME}/app" \
  "${APP_HOME}/config" \
  "${APP_HOME}/data/okf" \
  "${APP_HOME}/data/db"
install -m 0644 "${PROJECT_DIR}/src/okf_zvec.py" "${APP_HOME}/app/okf_zvec.py"

python3 -m venv "${APP_HOME}/.venv"
"${APP_HOME}/.venv/bin/pip" install --upgrade pip
"${APP_HOME}/.venv/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch
"${APP_HOME}/.venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"

if [[ ! -s "${APP_HOME}/config/service-token" ]]; then
  openssl rand -hex 32 > "${APP_HOME}/config/service-token"
  chmod 0600 "${APP_HOME}/config/service-token"
  echo "Токен синхронизации (сохраните его в защищённом месте):"
  cat "${APP_HOME}/config/service-token"
fi

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
