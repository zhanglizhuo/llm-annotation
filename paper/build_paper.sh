#!/usr/bin/env bash
#
# build_paper.sh — Compile IEEE Access manuscript(s)
#
# Usage:
#   bash build_paper.sh              # compile both versions
#   bash build_paper.sh revised      # revised manuscript only
#   bash build_paper.sh access       # original submission only
#
# Requires: pdflatex (TeX Live), pdfinfo (poppler-utils)
#
# The IEEE Access template ships custom fonts (t1-times, t1-formata,
# t1-giovannistd) stored in paper/ieeeaccess_template/. This script
# sets TEXINPUTS + TFMFONTS + T1FONTS + TEXFONTMAPS so that LaTeX
# finds them without slow font-substitution fallback.

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="$PAPER_DIR/ieeeaccess_template"

# ── Font / input search paths ────────────────────────────────────
TEXINPUTS="$TEMPLATE_DIR:"
TFMFONTS="$TEMPLATE_DIR:"
T1FONTS="$TEMPLATE_DIR:"
ENCFONTS="$TEMPLATE_DIR:"
TEXFONTMAPS="$TEMPLATE_DIR:"
export TEXINPUTS TFMFONTS T1FONTS ENCFONTS TEXFONTMAPS

# ── Which file(s) to compile ─────────────────────────────────────
TARGET="${1:-all}"

compile_one() {
    local tex="$1"
    local log="${tex%.tex}.log"
    local pdf="${tex%.tex}.pdf"

    echo "=== Compiling $tex ==="

    # Pass 1: generate aux / toc
    echo -n "  Pass 1... "
    pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1 || true
    echo "done"

    # Pass 2: resolve cross-references
    echo -n "  Pass 2... "
    pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1 || true
    echo "done"

    local errors
    errors=$(grep -c '^!' "$log" 2>/dev/null || echo 0)
    local pages
    pages=$(pdfinfo "$pdf" 2>/dev/null | awk '/Pages/ {print $2}')

    echo "  → $pdf  ($pages pages)"

    if [ "$errors" -gt 0 ]; then
        echo "  ⚠ $errors LaTeX errors in $log"
    fi
    echo
}

case "$TARGET" in
    all)
        compile_one "llm_annotation_paper_access.tex"
        compile_one "llm_annotation_paper_access_revised.tex"
        ;;
    access|original)
        compile_one "llm_annotation_paper_access.tex"
        ;;
    revised|rev)
        compile_one "llm_annotation_paper_access_revised.tex"
        ;;
    *)
        echo "Usage: $0 {all|revised|access}"
        exit 1
        ;;
esac

echo "Done."
