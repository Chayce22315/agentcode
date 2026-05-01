#!/usr/bin/env bash
# Packages a built .app into an unsigned IPA (Payload/MyApp.app -> zip).
set -euo pipefail

usage() {
  echo "Usage: $0 <path-to-app-bundle> [output-ipa-path]" >&2
  exit 1
}

[[ ${1:-} ]] || usage

APP_PATH="$1"
OUT_IPA="${2:-AIApp-unsigned.ipa}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "error: not a directory: $APP_PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

STAGING="$TMP_ROOT/staging"
mkdir -p "$STAGING/Payload"
cp -R "$APP_PATH" "$STAGING/Payload/"

(
  cd "$STAGING"
  zip -qr "$TMP_ROOT/ipa.zip" Payload
)

mv "$TMP_ROOT/ipa.zip" "$OUT_IPA"
echo "Wrote $OUT_IPA"
