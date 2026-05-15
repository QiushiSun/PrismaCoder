#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# VisPlotBench - one-shot environment setup (OPTIONAL convenience script)
#
# This script is NOT invoked by run.sh. It exists purely as a guided installer
# so users don't have to read the README to figure out what to install. You
# remain in control: nothing here touches your system without you running it.
#
# Usage:
#   ./setup.sh              # install everything (python deps + playwright + check tools)
#   ./setup.sh --py-only    # just `pip install -r requirements.txt`
#   ./setup.sh --check      # don't install anything, just report what's missing
# -----------------------------------------------------------------------------
set -euo pipefail

MODE="full"
case "${1:-}" in
    --py-only) MODE="py" ;;
    --check)   MODE="check" ;;
    "" )       MODE="full" ;;
    *) echo "Unknown arg: $1"; echo "Usage: $0 [--py-only|--check]"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

say() { printf "\n\033[1;34m== %s ==\033[0m\n" "$*"; }
ok()  { printf "  \033[32mok\033[0m   %s\n" "$*"; }
miss(){ printf "  \033[33mmiss\033[0m %s\n" "$*"; }
err() { printf "  \033[31mfail\033[0m %s\n" "$*"; }

# -----------------------------------------------------------------------------
# 1. Python packages
# -----------------------------------------------------------------------------
if [ "$MODE" = "full" ] || [ "$MODE" = "py" ]; then
    say "Installing Python deps from requirements.txt"
    python -m pip install -r requirements.txt
fi

# -----------------------------------------------------------------------------
# 2. Playwright browser (only needed if `playwright` is installed)
# -----------------------------------------------------------------------------
if [ "$MODE" = "full" ]; then
    if python -c "import playwright" 2>/dev/null; then
        say "Installing playwright chromium (needed for html/svg)"
        python -m playwright install chromium
    fi
fi

# -----------------------------------------------------------------------------
# 3. System-level tool check (we do NOT auto-install — too OS-specific)
# -----------------------------------------------------------------------------
say "Checking external CLI tools"
_check_tool() {
    local tool="$1"
    local hint="$2"
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool found at $(command -v "$tool")"
    else
        miss "$tool — $hint"
    fi
}
_check_tool asy      "asymptote      (macOS: brew install asymptote)"
_check_tool lualatex "latex          (macOS: install MacTeX; Linux: apt install texlive-luatex)"
_check_tool lilypond "lilypond       (macOS: brew install lilypond; Linux: apt install lilypond)"
_check_tool node     "mermaid/vegalite (macOS: brew install node)"
_check_tool mmdc     "mermaid CLI    (npm i -g @mermaid-js/mermaid-cli)"

say "Done."
echo "Next: export OPENAI_API_KEY=\"sk-...\"  then  ./run.sh --run-id 1"
