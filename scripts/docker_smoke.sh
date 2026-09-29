#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo 'Install Docker Engine/Desktop and Compose v2 first.' >&2; exit 1; }
docker compose version
docker compose config --quiet
docker compose up --build -d --wait
docker compose exec -T backend python -c "import urllib.request,json; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); assert data['status']=='ok'"
docker compose exec -T frontend wget -q -O - http://127.0.0.1:8080/api/health
echo '
Smoke check passed. The stack is still running; stop with docker compose down.'
