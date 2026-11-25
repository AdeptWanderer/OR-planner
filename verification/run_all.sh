#!/usr/bin/env bash
# Full verification run: results audit + every conformance test file + unit tests.
# Output: verification/reports/full_run.txt
cd "$(dirname "$0")/.."
R=verification/reports
{
echo "=== AUDIT $(date -u +%T)"
python3 verification/audit_results.py $R/paper_budget_results.jsonl $R/ckpt --crosscheck 120
echo "=== PRICING $(date -u +%T)"
python3 -m pytest -q -rA tests/conformance/test_s3_s4_pricing.py 2>&1 | grep -E "PASSED|FAILED|SKIPPED|ERROR|passed|failed|Error" | tail -90
echo "=== S2/S3/S5/ALG $(date -u +%T)"
python3 -m pytest -q -rA tests/conformance/test_s2_problem_definition.py tests/conformance/test_s3_master.py tests/conformance/test_s5_cuts_branching.py tests/conformance/test_algorithms_1_2.py 2>&1 | grep -E "PASSED|FAILED|SKIPPED|ERROR|passed|failed" | tail -60
echo "=== EXACTNESS $(date -u +%T)"
python3 -m pytest -q -rA tests/conformance/test_exactness.py 2>&1 | grep -E "PASSED|FAILED|SKIPPED|ERROR|passed|failed|assert" | tail -60
echo "=== UNIT $(date -u +%T)"
python3 -m pytest -q tests/test_orsched.py 2>&1 | tail -2
echo "=== END $(date -u +%T)"
} > $R/full_run.txt 2>&1
