#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <hosts_file> <config_path>"
  exit 1
fi

hosts_file="$1"
config_path="$2"
host_n=0

while read -r line <&3; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  node=$(echo "$line" | awk '{print $1}')

  ((host_n++))
  if (( host_n == 1 )); then
    continue
  fi

  remote_folder="~/surge/$(dirname "$config_path")"
  # echo "[$node] Creating remote folder: $remote_folder"
  ssh "$node" "mkdir -p $remote_folder"

  echo "[$node] Copying file to: ~/surge/$config_path"
  scp "$config_path" "$node:~/surge/$config_path"

done 3< "$hosts_file"

echo "Copied $config_path to all remote hosts."
