#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVER_SCRIPT="${SCRIPT_DIR}/run_ds_server_2gpu.sh"
VLLM_BIN="${SCRIPT_DIR}/.venv/bin/vllm"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
ENDPOINT="${ENDPOINT:-/v1/completions}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/ds_sweep}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/outputs/bench_ds_sweep}"
SUMMARY_PATH="${SUMMARY_PATH:-${SCRIPT_DIR}/deepseek_flash_random_input_sweep.md}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-1800}"
SERVER_SHUTDOWN_TIMEOUT="${SERVER_SHUTDOWN_TIMEOUT:-60}"
BENCH_REPETITIONS="${BENCH_REPETITIONS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-10}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

IFS=' ' read -r -a BENCH_CONFIGS <<< "${BENCH_CONFIGS:-baseline turbo_funnel_dense}"
IFS=' ' read -r -a INPUT_LENGTHS <<< "${INPUT_LENGTHS:-32768 131072 524288 1000000}"

SERVER_PID=""

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

cleanup() {
  stop_server || true
}

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    SERVER_PID=""
    return 0
  fi

  log "Stopping server pid=${SERVER_PID}"
  kill "${SERVER_PID}" 2>/dev/null || true

  local waited=0
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    if (( waited >= SERVER_SHUTDOWN_TIMEOUT )); then
      log "Server did not stop within ${SERVER_SHUTDOWN_TIMEOUT}s; sending SIGKILL"
      kill -9 "${SERVER_PID}" 2>/dev/null || true
      break
    fi
    sleep 1
    ((waited += 1))
  done

  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

wait_for_server() {
  local config="$1"
  local deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT))
  local models_url="${BASE_URL%/}/v1/models"

  log "Waiting for ${config} server readiness at ${models_url}"
  until curl -fsS "${models_url}" >/dev/null 2>&1; do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "Server exited before readiness check passed for ${config}"
      return 1
    fi
    if (( SECONDS >= deadline )); then
      log "Timed out waiting for server readiness after ${SERVER_STARTUP_TIMEOUT}s"
      return 1
    fi
    sleep 5
  done
}

start_server() {
  local config="$1"
  local launcher_log="${LOG_ROOT}/server_launcher_${config}.log"

  stop_server
  mkdir -p "${LOG_ROOT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN server start: bash ${SERVER_SCRIPT} ${config}"
    return 0
  fi

  log "Starting server for config=${config}"
  LOG_DIR="${LOG_ROOT}" PORT="${PORT}" bash "${SERVER_SCRIPT}" "${config}" >"${launcher_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${config}"
}

run_bench() {
  local config="$1"
  local input_len="$2"
  local repetition="$3"
  local run_dir="${RESULT_ROOT}/${config}/${input_len}/run${repetition}"
  local log_file="${run_dir}/bench.log"
  local result_json="${run_dir}/result.json"

  mkdir -p "${run_dir}"
  rm -f "${log_file}" "${result_json}" "${result_json%.json}.pytorch.json"

  local cmd=(
    "${VLLM_BIN}" bench serve
    --backend vllm
    --base-url "${BASE_URL}"
    --endpoint "${ENDPOINT}"
    --dataset-name random
    --random-input-len "${input_len}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --request-rate "${REQUEST_RATE}"
    --ignore-eos
    --seed "${SEED}"
    --save-result
    --save-detailed
    --result-dir "${run_dir}"
    --result-filename "$(basename "${result_json}")"
    --metadata "config=${config}" "input_len=${input_len}" "repetition=${repetition}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN bench: ${cmd[*]}"
    return 0
  fi

  log "Running benchmark config=${config} input_len=${input_len} repetition=${repetition}"
  "${cmd[@]}" 2>&1 | tee "${log_file}"

  if [[ ! -f "${result_json}" ]]; then
    echo "Expected result file was not created: ${result_json}" >&2
    return 1
  fi
}

write_summary() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN summary generation skipped"
    return 0
  fi

  "${PYTHON_BIN}" - "${RESULT_ROOT}" "${SUMMARY_PATH}" "${LOG_ROOT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

result_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
log_root = Path(sys.argv[3])


def pct_ms(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.array(samples, dtype=float), percentile) * 1000.0)


def mean_ms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return float(np.mean(np.array(samples, dtype=float)) * 1000.0)


def median_ms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return float(np.median(np.array(samples, dtype=float)) * 1000.0)


def fmt_float(value: float) -> str:
    return f"{value:.2f}"


rows = []
for result_json in sorted(result_root.glob("*/*/run3/result.json")):
    with result_json.open(encoding="utf-8") as fh:
        data = json.load(fh)

    config = str(data.get("config") or result_json.parents[2].name)
    input_len = int(data.get("input_len") or result_json.parents[1].name)
    ttfts = list(data.get("ttfts") or [])
    output_lens = list(data.get("output_lens") or [])
    itls_nested = list(data.get("itls") or [])

    flat_itls = [float(itl) for seq in itls_nested for itl in seq]
    tpots = []
    for output_len, seq in zip(output_lens, itls_nested):
        if output_len and output_len > 1:
            tpots.append(float(sum(seq)) / float(output_len - 1))

    rows.append(
        {
            "config": config,
            "input_len": input_len,
            "duration": float(data["duration"]),
            "request_throughput": float(data["request_throughput"]),
            "total_token_throughput": float(data["total_token_throughput"]),
            "failed": int(data.get("failed", 0)),
            "mean_ttft_ms": mean_ms(ttfts),
            "median_ttft_ms": median_ms(ttfts),
            "p99_ttft_ms": pct_ms(ttfts, 99),
            "mean_tpot_ms": mean_ms(tpots),
            "median_tpot_ms": median_ms(tpots),
            "p99_tpot_ms": pct_ms(tpots, 99),
            "mean_itl_ms": mean_ms(flat_itls),
            "median_itl_ms": median_ms(flat_itls),
            "p99_itl_ms": pct_ms(flat_itls, 99),
            "log_path": str((result_json.parent / "bench.log").resolve()),
            "result_path": str(result_json.resolve()),
        }
    )

rows.sort(key=lambda row: (row["config"], row["input_len"]))

server_logs = sorted(
    path for path in log_root.glob("server_*.log")
    if not path.name.startswith("server_launcher_")
)
generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as out:
    out.write("# DeepSeek Flash Random-Input Sweep\n\n")
    out.write(f"Generated: `{generated_at}`\n\n")
    out.write("Recorded result policy: keep all three runs for each configuration and input length, and report only `run3` below.\n\n")

    if server_logs:
        out.write("## Server Logs\n\n")
        for server_log in server_logs:
            out.write(f"- `{server_log.resolve()}`\n")
        out.write("\n")

    out.write("## Recorded Results\n\n")
    out.write(
        "| config | input_len | failed | duration_s | req/s | tok/s | "
        "mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | "
        "median_tpot_ms | p99_tpot_ms | mean_itl_ms | median_itl_ms | "
        "p99_itl_ms | run3 log |\n"
    )
    out.write(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )

    for row in rows:
        out.write(
            "| "
            f"{row['config']} | "
            f"{row['input_len']} | "
            f"{row['failed']} | "
            f"{fmt_float(row['duration'])} | "
            f"{fmt_float(row['request_throughput'])} | "
            f"{fmt_float(row['total_token_throughput'])} | "
            f"{fmt_float(row['mean_ttft_ms'])} | "
            f"{fmt_float(row['median_ttft_ms'])} | "
            f"{fmt_float(row['p99_ttft_ms'])} | "
            f"{fmt_float(row['mean_tpot_ms'])} | "
            f"{fmt_float(row['median_tpot_ms'])} | "
            f"{fmt_float(row['p99_tpot_ms'])} | "
            f"{fmt_float(row['mean_itl_ms'])} | "
            f"{fmt_float(row['median_itl_ms'])} | "
            f"{fmt_float(row['p99_itl_ms'])} | "
            f"`{row['log_path']}` |\n"
        )

    out.write("\n")
    out.write("## Raw Artifacts\n\n")
    out.write(f"- Benchmark results root: `{result_root.resolve()}`\n")
    out.write("- Each `(config, input_len)` directory contains `run1`, `run2`, and `run3` with raw logs and saved JSON results.\n")

if not rows:
    raise SystemExit(f"No run3 result files found under {result_root}")
PY

  log "Summary written to ${SUMMARY_PATH}"
}

trap cleanup EXIT

mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

for config in "${BENCH_CONFIGS[@]}"; do
  start_server "${config}"
  for input_len in "${INPUT_LENGTHS[@]}"; do
    for repetition in $(seq 1 "${BENCH_REPETITIONS}"); do
      run_bench "${config}" "${input_len}" "${repetition}"
    done
  done
  stop_server
done

write_summary
