#!/usr/bin/env bash
# AGENT INSTRUCTION: dual-Colab-account switcher. Source it, don't exec it:
#   source /media/pope/projecteo/connect/connect_gpu/use-colab.sh pro   # Pro account (default)
#   source /media/pope/projecteo/connect/connect_gpu/use-colab.sh old   # old free-tier account
#   source /media/pope/projecteo/connect/connect_gpu/use-colab.sh show  # which account is live
# After sourcing, run colab via:  colab --config "$COLAB_CONFIG" <cmd>
# The CLI hardcodes its grant at ~/.config/colab-cli/token.json, so switching
# accounts = swapping which stored grant occupies that slot. Stored grants stay
# machine-local in HOME (0600) and NEVER move to shared/project storage.
set -uo pipefail

_COLAB_DIR="${HOME}/.config/colab-cli"
_COLAB_LIVE="${_COLAB_DIR}/token.json"
_COLAB_PRO="${_COLAB_DIR}/token.pro.json"
_COLAB_OLD="${_COLAB_DIR}/token.old.json"

_colab_hash() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }

use-colab-show() {
  local live pro old
  live="$(_colab_hash "$_COLAB_LIVE")"; pro="$(_colab_hash "$_COLAB_PRO")"; old="$(_colab_hash "$_COLAB_OLD")"
  if [ -n "$live" ] && [ "$live" = "$pro" ]; then echo "live colab account: pro"
  elif [ -n "$live" ] && [ "$live" = "$old" ]; then echo "live colab account: old"
  else echo "live colab account: UNKNOWN (token.json matches neither stored grant)" >&2; return 1
  fi
}

use-colab-switch() {
  local acct="${1:?usage: use-colab-switch pro|old}" src cfg
  case "$acct" in
    pro) src="$_COLAB_PRO"; cfg="${_COLAB_DIR}/sessions.pro.json" ;;
    old) src="$_COLAB_OLD"; cfg="${_COLAB_DIR}/sessions.old.json" ;;
    *) echo "unknown colab account: $acct (pro|old)" >&2; return 1 ;;
  esac
  [ -f "$src" ] || { echo "missing grant file: $src" >&2; return 1; }
  cp -f "$src" "$_COLAB_LIVE" && chmod 600 "$_COLAB_LIVE"
  export COLAB_ACCOUNT="$acct" COLAB_CONFIG="$cfg"
  echo "colab account -> $acct (config: $cfg)"
}

case "${1:-pro}" in
  show) use-colab-show ;;
  pro|old) use-colab-switch "$1" ;;
  *) echo "usage: source use-colab.sh [pro|old|show]" >&2; return 1 2>/dev/null || exit 1 ;;
esac
