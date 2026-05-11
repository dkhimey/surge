#!/bin/bash

# Usage: ./generate_rankfile.sh hosts.txt 28

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <hosts_file> <procs_per_node> <cores_per_node>"
  exit 1
fi

hosts_file="$1"
cores_per_node="$2"
rank=1
host_n=0
end_core=$(( $3 - 1 ))

# Clear or create rankfile.txt
> rankfile.txt

# Read hostnames (first word of each non-empty line)
while read -r line; do
  # Skip empty or comment lines
  [[ -z "$line" || "$line" =~ ^# ]] && continue

  # Extract hostname (first field before space)
  node=$(echo "$line" | awk '{print $1}')

  if (( $host_n == 0 )) 
  then
    echo "rank 0=$node slot=0:0-$end_core" >> rankfile.txt
  else
    for ((core=0; core<cores_per_node; core++)); do
        echo "rank $rank=$node slot=$core:0-$end_core" >> rankfile.txt
        ((rank++))
    done
  fi

  ((host_n++));
done < "$hosts_file"

echo "Generated rankfile.txt with $rank total ranks."