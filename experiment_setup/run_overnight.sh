#!/usr/bin/env bash
# run_overnight.sh
#
# Runs shared_batch_experiment_sweep for all three MSTuring-100M runbook
# types (clustered, random, shift) at two maintenance thresholds:
#   t=6000  → maintenance enabled  (τ = 0.6, 60% of 10 000 centers)
#   t=10000 → maintenance disabled (≥ NCENTERS, never triggers)
#
# 6 runs total, executed sequentially.  Each run is identified solely by
# (dataset, threshold); all output paths are derived from that identity so
# that a restarted run automatically resumes and appends to the same files:
#
#   results/<dataset>_t<threshold>/results.csv   ← append on resume
#   results/<dataset>_t<threshold>/run.log       ← append on resume
#   checkpoints/<dataset>_t<threshold>/          ← managed by the binary
#
# Prerequisites:
#   1. Build:  make experiments
#   2. A rankfile.txt in the working directory
#      (see experiment_setup/generate_rankfile.sh)
#
# Usage:  ./run_overnight.sh [rankfile]
#   rankfile defaults to ./rankfile.txt

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
BINARY="./bin/shared_batch_experiment_sweep"
NUM_PARTITIONS=10
NUM_RANKS=$(( NUM_PARTITIONS + 1 ))   # 11: 1 coordinator + 10 executors
K=10
RANKFILE="${1:-rankfile.txt}"

# MPI transport: restrict to the flat LAN interface so Open MPI doesn't
# accidentally route inter-node traffic over the wrong interface on CloudLab.
MCA_OPTS="-mca btl_tcp_if_include br-flat-lan-1"

# CPU pinning: bind each rank to physical cores 0-31.
TASKSET="taskset -c 0-31"

# Maintenance thresholds to sweep
THRESHOLDS=(6000 10000)

# ── Dataset registry ──────────────────────────────────────────────────────────
# Format: "dataset_key|gt_prefix|runbook_yaml"
declare -a DATASETS=(
    "msturing-100M-clustered|/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100000000/msturing-100M-clustered_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/msturing-100M-clustered_runbookfinal.yaml"
#    "msturing-100M-random|/dataset/big-ann-benchmarks/data/MSTuring-100M-random/100000000/runbook-msturing-100M-random.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-random/runbook-msturing-100M-random.yaml"
    "msturing-100M-shift|/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100000000/msturing-100M-shift_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/msturing-100M-shift_runbookfinal.yaml"
    "bigann-100M-clustered|/dataset/big-ann-benchmarks/data/bigann-clustered/100000000/runbook-bigann-100M.yaml|/dataset/big-ann-benchmarks/data/bigann-clustered/runbook-bigann-100M.yaml"
#    "bigann-100M-random|/dataset/big-ann-benchmarks/data/bigann-random/100000000/runbook-bigann-100M-random.yaml|/dataset/big-ann-benchmarks/data/bigann-random/runbook-bigann-100M-random.yaml"
    "bigann-100M-shift|/dataset/big-ann-benchmarks/data/bigann-shift/100000000/bigann-100M-shift_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/bigann-shift/bigann-100M-shift_runbookfinal.yaml"
)

# ── Helper: last step number in a runbook YAML ────────────────────────────────
runbook_last_step() {
    local runbook="$1" dataset_key="$2"
    python3 - "$runbook" "$dataset_key" <<'EOF'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
try:
    doc  = yaml.safe_load(open(path))
    sub  = doc.get(key, {})
    nums = [int(k) for k in sub if str(k).isdigit()]
    print(max(nums) if nums else -1)
except Exception:
    print(-1)
EOF
}

# ── Helper: true if CSV exists and contains the last runbook step ─────────────
is_complete() {
    local csv="$1" runbook="$2" dataset_key="$3"
    [[ -f "$csv" ]] || return 1
    local last_step
    last_step=$(runbook_last_step "$runbook" "$dataset_key")
    [[ "$last_step" -gt 0 ]] && grep -q "^${last_step}," "$csv"
}

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY — run 'make experiments' first"
    exit 1
fi

if [ ! -f "$RANKFILE" ]; then
    echo "ERROR: rankfile not found at $RANKFILE"
    echo "       Generate one with: experiment_setup/generate_rankfile.sh <hosts.txt> <procs_per_node> <cores_per_node>"
    exit 1
fi

echo "Starting at $(date)"
echo

# ── Run loop ──────────────────────────────────────────────────────────────────
FAILED=()

for entry in "${DATASETS[@]}"; do
    DATASET="${entry%%|*}"
    _rest="${entry#*|}"
    GT_PREFIX="${_rest%%|*}"
    RUNBOOK="${_rest##*|}"

    for THRESHOLD in "${THRESHOLDS[@]}"; do
        TAG="${DATASET}_t${THRESHOLD}"
        RUN_DIR="results/${TAG}"
        OUTPUT="${RUN_DIR}/results.csv"
        LOG="${RUN_DIR}/run.log"

        mkdir -p "$RUN_DIR"

        if is_complete "$OUTPUT" "$RUNBOOK" "$DATASET"; then
            echo "════════════════════════════════════════════════════════════"
            echo "  dataset   : $DATASET"
            echo "  threshold : $THRESHOLD"
            echo "  → already complete, skipping"
            echo "════════════════════════════════════════════════════════════"
            echo
            continue
        fi

        echo "════════════════════════════════════════════════════════════"
        echo "  dataset   : $DATASET"
        echo "  threshold : $THRESHOLD"
        echo "  output    : $OUTPUT"
        echo "  log       : $LOG"
        echo "  started   : $(date)"
        echo "════════════════════════════════════════════════════════════"

        # Append a run-attempt header to the log (>> so prior attempts are kept).
        {
            echo "──────────────────────────────────────────────"
            echo "attempt   : $(date)"
            echo "dataset   : $DATASET"
            echo "threshold : $THRESHOLD"
            echo "gt_prefix : $GT_PREFIX"
            echo "output    : $OUTPUT"
            echo "command   : mpirun $MCA_OPTS -np $NUM_RANKS --rankfile $RANKFILE $TASKSET $BINARY $DATASET $NUM_PARTITIONS $THRESHOLD $K $GT_PREFIX $OUTPUT"
            echo "---"
        } >> "$LOG"

        # shellcheck disable=SC2086
        if mpirun $MCA_OPTS -np "$NUM_RANKS" --rankfile "$RANKFILE" \
               $TASKSET "$BINARY" \
               "$DATASET" "$NUM_PARTITIONS" "$THRESHOLD" "$K" \
               "$GT_PREFIX" "$OUTPUT" \
               2>&1 | tee -a "$LOG"; then
            echo "  → finished at $(date)"
        else
            EXIT_CODE=${PIPESTATUS[0]}
            echo "  → FAILED at $(date) (exit $EXIT_CODE)"
            FAILED+=("$TAG")
        fi

        echo
        sleep 10   # let MPI daemons tear down before the next run
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "All runs complete at $(date)"
echo

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs succeeded."
else
    echo "FAILED runs (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
