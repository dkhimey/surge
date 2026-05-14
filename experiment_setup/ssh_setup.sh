#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <hosts_file>"
  exit 1
fi

hosts_file="$1"
host_n=0

# Read hostnames (first word of each non-empty line)
while read -r line; do
  # Skip empty or comment lines
  [[ -z "$line" || "$line" =~ ^# ]] && continue

  # Extract hostname (first field before space)
  node=$(echo "$line" | awk '{print $1}')


  ssh-keyscan "$node" >> ~/.ssh/known_hosts

  ((host_n++));
done < "$hosts_file"

echo "Added all executors to known hosts file."
