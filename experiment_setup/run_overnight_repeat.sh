#!/usr/bin/env bash
# run_overnight.sh
#
# Runs shared_batch_experiment_sweep for all three MSTuring-100M runbook
# types (clustered, random, shift) at two maintenance thresholds:
#   t=6000  → maintenance enabled  (τ = 0.6, 60% of 10 000 centers)
#   t=10000 → maintenance disabled (≥ NCENTERS, never triggers)
#
# Each sweep starts from the cluster-analysis step-1 state for its dataset
# (--init-state-dir sim-<dataset> --init-partitions sim-<dataset>/step_000001_partitions.csv)
# instead of building the index from scratch.
#
# Searches are run at several target fractions of the (insert+delete+search)
# vector workload in one pass via --search-fraction (see SEARCH_FRACTION below);
# insert/delete steps are unchanged and recall is computed over the original
# query set.
#
# Runs are executed sequentially.  Each run is identified solely by
# (dataset, threshold); all output paths are derived from that identity so
# that a restarted run automatically resumes and appends to the same files:
#
#   results/<dataset>_t<threshold>/results.csv   ← append on resume
#   results/<dataset>_t<threshold>/run.log       ← append on resume
#   checkpoints/<dataset>_t<threshold>/          ← managed by the binary
#
# Note: the starting state is used only on a fresh start; if a checkpoint
# exists for a (dataset, threshold) the binary resumes from it and ignores
# the --init-state-* flags.
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

# Number of times to attempt each MPI run before giving up.  Because the binary
# resumes from its checkpoint, a retry continues where the failed attempt left
# off rather than starting over.
MAX_ATTEMPTS=3

# Search-query scaling: run each search step at MULTIPLE target fractions of the
# (insert+delete+search) vector workload in a single pass.  The binary tiles each
# search step's query batch by the factor needed to hit each fraction (insert/
# delete steps are left unchanged); QPS is measured over the scaled query count
# while recall is computed once over the original query set.  Every fraction is
# measured on the IDENTICAL index state, so the rows can be compared directly via
# the search_fraction CSV column.  Comma-separated list; empty disables scaling.
SEARCH_FRACTION=

# MPI transport: restrict to the flat LAN interface so Open MPI doesn't
# accidentally route inter-node traffic over the wrong interface on CloudLab.
MCA_OPTS="-mca btl_tcp_if_include br-flat-lan-1"

# CPU pinning: bind each rank to physical cores 0-31.
TASKSET="taskset -c 0-31"

# Maintenance thresholds to sweep
THRESHOLDS=(10000)

# Starting state: each sweep begins from the cluster-analysis step-1 output
# instead of building from scratch.  For dataset <key> the state lives in
# sim-<key>/ (meta-HNSW, centroids, counts, label→center) with the precomputed
# center→shard partitions at sim-<key>/step_000001_partitions.csv.
INIT_STATE_STEP=1

# ── Dataset registry ──────────────────────────────────────────────────────────
# Format: "dataset_key|gt_prefix|runbook_yaml"
declare -a DATASETS=(
#    "msturing-100M-clustered|/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100000000/msturing-100M-clustered_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/msturing-100M-clustered_runbookfinal.yaml"
#    "msturing-100M-random|/dataset/big-ann-benchmarks/data/MSTuring-100M-random/100000000/runbook-msturing-100M-random.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-random/runbook-msturing-100M-random.yaml"
#    "msturing-100M-shift|/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100000000/msturing-100M-shift_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/msturing-100M-shift_runbookfinal.yaml"
#    "msturing-500M-shift|/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/500000000/runbook_msturing500Mshift.yaml|/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/runbook_msturing500Mshift.yaml"
#    "msturing-500M-clustered|/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/500000000/runbook_msturing500Mclustered.yaml|/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/runbook_msturing500Mclustered.yaml"
#    "bigann-100M-clustered|/dataset/big-ann-benchmarks/data/bigann-clustered/100000000/runbook-bigann-100M.yaml|/dataset/big-ann-benchmarks/data/bigann-clustered/runbook-bigann-100M.yaml"
#    "bigann-100M-random|/dataset/big-ann-benchmarks/data/bigann-random/100000000/runbook-bigann-100M-random.yaml|/dataset/big-ann-benchmarks/data/bigann-random/runbook-bigann-100M-random.yaml"
#   "bigann-100M-shift|/dataset/big-ann-benchmarks/data/bigann-shift/100000000/bigann-100M-shift_runbookfinal.yaml|/dataset/big-ann-benchmarks/data/bigann-shift/bigann-100M-shift_runbookfinal.yaml"
#    "bigann-500M-clustered|/dataset/big-ann-benchmarks/data/bigann-500M-clustered/500000000/runbook_bigann-500M-clustered.yaml|/dataset/big-ann-benchmarks/data/bigann-500M-clustered/runbook_bigann-500M-clustered.yaml"
     "bigann-500M-shift|/dataset/big-ann-benchmarks/data/bigann-500M-shift/500000000/runbook_bigann500Mshift.yaml|/dataset/big-ann-benchmarks/data/bigann-500M-shift/runbook_bigann500Mshift.yaml"
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

    # Step-1 starting state produced by msturing-cluster-analysis.cpp for this
    # dataset (e.g. sim-msturing-100M-clustered) plus its precomputed partitions.
    INIT_STATE_DIR="sim-${DATASET}"
    INIT_PARTITIONS="${INIT_STATE_DIR}/step_000001_partitions.csv"
    STEP_PREFIX="$(printf 'step_%06d' "$INIT_STATE_STEP")"

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

        # Verify the step-1 starting-state files exist before launching.  These
        # are ignored on resume (a checkpoint takes precedence), but a fresh run
        # needs them, so fail fast with a clear message if they are missing.
        if [[ ! -f "${INIT_STATE_DIR}/${STEP_PREFIX}_hnsw.bin" || ! -f "$INIT_PARTITIONS" ]]; then
            echo "════════════════════════════════════════════════════════════"
            echo "  dataset   : $DATASET"
            echo "  threshold : $THRESHOLD"
            echo "  → ERROR: starting-state files missing; skipping"
            echo "           expected ${INIT_STATE_DIR}/${STEP_PREFIX}_hnsw.bin"
            echo "           and      $INIT_PARTITIONS"
            echo "════════════════════════════════════════════════════════════"
            echo
            FAILED+=("$TAG (missing starting state)")
            continue
        fi

        echo "════════════════════════════════════════════════════════════"
        echo "  dataset   : $DATASET"
        echo "  threshold : $THRESHOLD"
        echo "  init state: $INIT_STATE_DIR (step $INIT_STATE_STEP)"
        echo "  partitions: $INIT_PARTITIONS"
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
            echo "init state: $INIT_STATE_DIR (step $INIT_STATE_STEP)"
            echo "partitions: $INIT_PARTITIONS"
            echo "output    : $OUTPUT"
            echo "command   : mpirun $MCA_OPTS -np $NUM_RANKS --rankfile $RANKFILE $TASKSET $BINARY $DATASET $NUM_PARTITIONS $THRESHOLD $K $GT_PREFIX $OUTPUT --init-state-dir $INIT_STATE_DIR --init-partitions $INIT_PARTITIONS --init-state-step $INIT_STATE_STEP${SEARCH_FRACTION:+ --search-fraction $SEARCH_FRACTION}"
            echo "---"
        } >> "$LOG"

        # Run the MPI command, retrying up to MAX_ATTEMPTS times on failure.
        # Because the binary resumes from its checkpoint, each retry continues
        # where the failed attempt left off rather than starting over.
        SUCCEEDED=0
        for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
            echo "  → attempt $ATTEMPT of $MAX_ATTEMPTS at $(date)"

            # shellcheck disable=SC2086
            if mpirun $MCA_OPTS -np "$NUM_RANKS" --rankfile "$RANKFILE" \
                   $TASKSET "$BINARY" \
                   "$DATASET" "$NUM_PARTITIONS" "$THRESHOLD" "$K" \
                   "$GT_PREFIX" "$OUTPUT" \
                   --init-state-dir "$INIT_STATE_DIR" \
                   --init-partitions "$INIT_PARTITIONS" \
                   --init-state-step "$INIT_STATE_STEP" \
                   ${SEARCH_FRACTION:+--search-fraction "$SEARCH_FRACTION"} \
                   2>&1 | tee -a "$LOG"; then
                echo "  → finished at $(date) (attempt $ATTEMPT)"
                SUCCEEDED=1
                break
            else
                EXIT_CODE=${PIPESTATUS[0]}
                echo "  → attempt $ATTEMPT FAILED at $(date) (exit $EXIT_CODE)"
                if [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
                    echo "  → retrying after 10s..."
                    sleep 10
                fi
            fi
        done

        if [ "$SUCCEEDED" -ne 1 ]; then
            echo "  → all $MAX_ATTEMPTS attempts FAILED for $TAG"
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
