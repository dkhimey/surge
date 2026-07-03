#!/bin/bash
 
set -e
 
declare -A EXPERIMENTS
 
#EXPERIMENTS["bigann-random"]="
#    --base-dir       /dataset/big-ann-benchmarks/data/bigann-random/cluster_history_bigann-100M-random_10000
#    --partitions-dir /dataset/big-ann-benchmarks/data/bigann-random/cluster_history_bigann-100M-random_10000_10
#    --runbook-path   /dataset/big-ann-benchmarks/data/bigann-random/runbook-bigann-100M-random.yaml
#    --query-file     /dataset/big-ann-benchmarks/data/bigann-random/query.public.10K.u8bin
#    --base-file      /dataset/big-ann-benchmarks/data/bigann-random/bigann-100M-random.u8bin
#    --gt-dir         /dataset/big-ann-benchmarks/data/bigann-random/100000000/runbook-bigann-100M-random.yaml
#"
 
#EXPERIMENTS["bigann-clustered"]="
#    --base-dir       /dataset/big-ann-benchmarks/data/bigann-clustered/cluster_history_bigann-100M-clustered_10000
#    --partitions-dir /dataset/big-ann-benchmarks/data/bigann-clustered/cluster_history_bigann-100M-clustered_10000_10
#    --runbook-path   /dataset/big-ann-benchmarks/data/bigann-clustered/runbook-bigann-100M.yaml
#    --query-file     /dataset/big-ann-benchmarks/data/bigann-clustered/query.public.10K.u8bin
#    --base-file      /dataset/big-ann-benchmarks/data/bigann-clustered/bigann-100M-clustered.u8bin
#    --gt-dir         /dataset/big-ann-benchmarks/data/bigann-clustered/100000000/runbook-bigann-100M.yaml/
#"
 
#EXPERIMENTS["msturing-clustered"]="
#    --base-dir       /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/cluster_history_msturing-100M-clustered_10000
#    --partitions-dir /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/cluster_history_msturing-100M-clustered_10000_10
#    --runbook-path   /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/msturing-100M-clustered_runbookfinal.yaml
#    --query-file     /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/testQuery10K.fbin
#    --base-file      /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100M-msturing-clustered.fbin
#    --gt-dir         /dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100000000/msturing-100M-clustered_runbookfinal.yaml/
#"
 
#EXPERIMENTS["msturing-random"]="
#    --base-dir       /dataset/big-ann-benchmarks/data/MSTuring-100M-random/cluster_history_msturing-100M-random_10000
#    --partitions-dir /dataset/big-ann-benchmarks/data/MSTuring-100M-random/cluster_history_msturing-100M-random_10000_10/
#    --runbook-path   /dataset/big-ann-benchmarks/data/MSTuring-100M-random/runbook-msturing-100M-random.yaml
#    --query-file     /dataset/big-ann-benchmarks/data/MSTuring-100M-random/testQuery10K.fbin
#    --base-file      /dataset/big-ann-benchmarks/data/MSTuring-100M-random/msturing-100M-random.fbin
#    --gt-dir         /dataset/big-ann-benchmarks/data/MSTuring-100M-random/100000000/runbook-msturing-100M-random.yaml/
#"
 
#EXPERIMENTS["msturing-shift"]="
#    --base-dir       /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/cluster_history_msturing-100M-shift_10000
#    --partitions-dir /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/cluster_history_msturing-100M-shift_10000_10/
#    --runbook-path   /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/msturing-100M-shift_runbookfinal.yaml
#    --query-file     /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/testQuery10K.fbin
#    --base-file      /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100M-msturing-shift.fbin
#    --gt-dir         /dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100000000/msturing-100M-shift_runbookfinal.yaml/
#"

EXPERIMENTS["bigann-500M-clustered"]="
    --base-dir /users/dkhimey/extra/cluster_history_bigann500Mclustered
    --partitions-dir /users/dkhimey/extra/cluster_history_bigann500Mclustered
    --runbook-path /dataset/big-ann-benchmarks/data/bigann-500M-clustered/runbook_bigann-500M-clustered.yaml
    --query-file /dataset/big-ann-benchmarks/data/bigann-500M-clustered/query.public.10K.u8bin
    --base-file /dataset/big-ann-benchmarks/data/bigann-500M-clustered/500M-bigann64clustered.u8bin
    --gt-dir /dataset/big-ann-benchmarks/data/bigann-500M-clustered/500000000/runbook_bigann-500M-clustered.yaml
"

for experiment in "${!EXPERIMENTS[@]}"; do
    for threshold in 0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo "========================================="
        echo "Experiment: ${experiment}  threshold: ${threshold}"
        echo "========================================="
        sudo python -u compute_theoretical_recall_updated.py \
            ${EXPERIMENTS[$experiment]} \
            --mode NProbe \
            --threshold "$threshold"
    done
done
 
echo "Done."
