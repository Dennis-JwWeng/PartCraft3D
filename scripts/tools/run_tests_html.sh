#!/usr/bin/env bash
# run_tests_html.sh — Run the full PartCraft3D test suite and produce an HTML report.
#
# Usage:
#   bash scripts/tools/run_tests_html.sh [REPORT_DIR] [EXTRA_PYTEST_ARGS...]
#
# Examples:
#   bash scripts/tools/run_tests_html.sh
#   bash scripts/tools/run_tests_html.sh /tmp/test_report
#   bash scripts/tools/run_tests_html.sh /tmp/test_report -k "smoke"
#   bash scripts/tools/run_tests_html.sh /tmp/test_report -x        # stop on first failure
#
# Output:
#   {REPORT_DIR}/report.html   — self-contained, open in any browser
#   {REPORT_DIR}/pytest_log.jsonl — machine-readable log per test
#
# The script auto-detects the pipeline_server conda environment where
# openai/cv2/etc. are installed. Override with PYTHON env var:
#   PYTHON=/path/to/python bash scripts/tools/run_tests_html.sh
set -euo pipefail

# ─── config ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${1:-${REPO_ROOT}/test_reports}"
shift 1 2>/dev/null || true   # consume REPORT_DIR arg; remaining → pytest

REPORT_FILE="${REPORT_DIR}/report.html"
TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"

# ─── auto-detect Python with openai installed ─────────────────────────
if [ -z "${PYTHON:-}" ]; then
    for candidate in \
        /root/miniconda3/envs/pipeline_server/bin/python \
        /mnt/zsn/miniconda3/envs/pipeline_server/bin/python \
        python3 python; do
        if command -v "$candidate" &>/dev/null && \
           "$candidate" -c "import openai" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi
PYTHON="${PYTHON:-python3}"

# ─── ensure pytest-html is available ─────────────────────────────────
if ! "${PYTHON}" -c "import pytest_html" 2>/dev/null; then
    echo "[run_tests_html] Installing pytest-html into $(${PYTHON} -c 'import sys; print(sys.prefix)') ..."
    "${PYTHON}" -m pip install pytest-html --quiet
fi

mkdir -p "${REPORT_DIR}"

echo "============================================================"
echo " PartCraft3D Test Suite"
echo " Python    : ${PYTHON}"
echo " Timestamp : ${TIMESTAMP}"
echo " Report    : ${REPORT_FILE}"
echo " Tests dir : ${REPO_ROOT}/tests"
echo "============================================================"

cd "${REPO_ROOT}"

# ─── run tests ───────────────────────────────────────────────────────
set +e
"${PYTHON}" -m pytest tests/ \
    --html="${REPORT_FILE}" \
    --self-contained-html \
    --tb=short \
    -v \
    "$@"
EXIT_CODE=$?
set -e

# ─── summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo " RESULT: ALL TESTS PASSED  ✓"
else
    echo " RESULT: SOME TESTS FAILED  (exit code: ${EXIT_CODE})"
fi
echo ""
echo " HTML report saved to:"
echo "   ${REPORT_FILE}"
echo ""
echo " To download and view locally:"
echo "   scp <user>@<host>:${REPORT_FILE} ~/Downloads/partcraft3d_report.html"
echo "   open ~/Downloads/partcraft3d_report.html"
echo "============================================================"

exit ${EXIT_CODE}
