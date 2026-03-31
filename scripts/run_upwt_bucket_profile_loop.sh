#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/sglang"
LOG_PATH="/root/logs/perf_upwt_detailed.log"
TEST_TARGET="python/sglang/multimodal_gen/test/server/test_update_weights_from_tensor.py::TestUpdateWeightsFromTensor::test_update_weights_from_tensor_flattened_bucket_full_transformer"

# 512MB, 1G, 2G, 4G, 8G, 20G (bytes)
SIZES=(
  "536870912:512MB"
  "1073741824:1G"
  "2147483648:2G"
  "4294967296:4G"
  "8589934592:8G"
  "21474836480:20G"
)

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

echo "===== bucket profiling loop start: $(date -u +"%Y-%m-%dT%H:%M:%SZ") =====" >> "$LOG_PATH"
echo "test_target=$TEST_TARGET" >> "$LOG_PATH"

cd "$ROOT_DIR"

for entry in "${SIZES[@]}"; do
  IFS=":" read -r bucket_size_bytes bucket_label <<< "$entry"

  {
    echo ""
    echo "===== run start: bucket=${bucket_label} bytes=${bucket_size_bytes} utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
  } >> "$LOG_PATH"

  # Ensure server is restarted so env vars are picked up by worker processes.
  pkill -f "sglang" >/dev/null 2>&1 || true

  export SGLANG_DIFFUSION_PROFILE_UPDATE_TENSOR=1
  export SGLANG_DIFFUSION_UPDATE_TENSOR_PROFILE_LOG_PATH="$LOG_PATH"
  export SGLANG_MMGEN_TENSOR_BUCKET_SIZE_BYTES="$bucket_size_bytes"

  # Keep test output concise; detailed profile lines are appended to $LOG_PATH.
  if pytest -s -v "$TEST_TARGET"; then
    echo "===== run done: bucket=${bucket_label} status=PASS utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ") =====" >> "$LOG_PATH"
  else
    echo "===== run done: bucket=${bucket_label} status=FAIL utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ") =====" >> "$LOG_PATH"
    exit 1
  fi
done

echo "===== bucket profiling loop done: $(date -u +"%Y-%m-%dT%H:%M:%SZ") =====" >> "$LOG_PATH"
echo "all results appended to $LOG_PATH"
