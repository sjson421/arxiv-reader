#!/usr/bin/env bash
# Deploys origin/main into this checkout: fast-forward, test, rebuild the frontend, restart the web service.
# Runs from the self-hosted Actions runner (see .github/workflows/deploy.yml); safe to run by hand too.
# If the tests fail, the checkout goes back to the previous commit and the service is left alone.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Working tree has local changes, not deploying." >&2
    exit 1
fi

prev=$(git rev-parse HEAD)
git fetch --quiet origin main
git merge --ff-only origin/main
new=$(git rev-parse HEAD)
if [ "$prev" = "$new" ]; then
    echo "Already at ${new:0:7}."
    exit 0
fi

if ! { uv sync --quiet && uv run pytest -q; }; then
    echo "Tests failed on ${new:0:7}; going back to ${prev:0:7}." >&2
    git reset --hard "$prev"
    uv sync --quiet
    exit 1
fi

if ! git diff --quiet "$prev" "$new" -- frontend; then
    (cd frontend && npm ci --silent && npm run build)
fi
if ! git diff --quiet "$prev" "$new" -- systemd; then
    echo "systemd/ changed; copy the units to ~/.config/systemd/user and daemon-reload by hand." >&2
fi
systemctl --user restart arxiv-reader-web
echo "Deployed ${new:0:7}."
