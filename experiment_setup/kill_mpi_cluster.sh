#!/bin/bash
# kill_mpi_cluster.sh
#
# Kills all shared_batch_experiment_sweep processes and MPI daemons (orted)
# on every node in the cluster, then removes temp files left by the binary.
#
# Usage: ./kill_mpi_cluster.sh <hosts_file>
#
# hosts_file: same format used by generate_rankfile.sh — one hostname per
# line, first line is the coordinator (ctl), remaining lines are workers.
 
set -euo pipefail
 
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <hosts_file>"
    exit 1
fi
 
HOSTS_FILE="$1"
BINARY_NAME="shared_batch_experiment_sweep"
WORK_DIR="~/surge"   # remote working directory where temp files are written
 
# Collect all nodes (coordinator + workers)
mapfile -t NODES < <(awk 'NF && !/^#/ {print $1}' "$HOSTS_FILE")
 
if [ ${#NODES[@]} -eq 0 ]; then
    echo "ERROR: no hosts found in $HOSTS_FILE"
    exit 1
fi
 
echo "Killing $BINARY_NAME and orted on ${#NODES[@]} nodes..."
echo
 
for node in "${NODES[@]}"; do
    echo -n "  $node ... "
    ssh -o ConnectTimeout=5 "$node" bash <<EOF
# Kill the experiment binary (all instances)
pkill -9 -f "$BINARY_NAME" 2>/dev/null && echo -n "binary killed " || echo -n "binary not running "
 
# Kill Open MPI daemons
pkill -9 orted 2>/dev/null && echo -n "orted killed " || echo -n "orted not running "
 
# Remove temp files written by the binary (broadcast buffers)
rm -f $WORK_DIR/tmp_shared_sweep*.bin && echo -n "tmp files removed " || true
 
echo ""
EOF
    echo "done"
done
 
echo
echo "All nodes cleaned up."
