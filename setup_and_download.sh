#!/usr/bin/env bash
# =============================================================================
#  setup_and_download.sh
#  1. Creates (or updates) the 'copernicus' conda environment
#  2. Activates it
#  3. Downloads GLORYS 1995 SST + SSH into ./data/
#
#  Usage (from the phase1 folder):
#      bash setup_and_download.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Locate conda ───────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    # Try common install locations
    for p in ~/miniconda3/bin/conda ~/anaconda3/bin/conda \
              /opt/homebrew/Caskroom/miniconda/base/bin/conda \
              /opt/miniconda3/bin/conda; do
        [ -f "$p" ] && { source "$(dirname "$p")/../etc/profile.d/conda.sh"; break; }
    done
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda and re-run."
    exit 1
fi

# ── 2. Create or update the environment ──────────────────────────────────────
echo "==> Creating/updating conda environment 'copernicus'..."
if conda env list | grep -q "^copernicus "; then
    echo "    Environment already exists – updating from copernicus.yml"
    conda env update --name copernicus --file copernicus.yml --prune
else
    conda env create --file copernicus.yml
fi

# ── 3. Activate and download ──────────────────────────────────────────────────
echo "==> Activating environment and downloading GLORYS data..."
conda run -n copernicus python download_glorys_1995.py

echo ""
echo "Done! Files are in: $SCRIPT_DIR/data/"
ls -lh "$SCRIPT_DIR/data/" 2>/dev/null || true
