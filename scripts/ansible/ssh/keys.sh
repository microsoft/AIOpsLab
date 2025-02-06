#!/bin/bash

WORKERS_FILE="hosts.txt"

# Read the file line by line, 
# including the last line 
# even if it doesn't have a newline
while IFS= read -r host_ip || [[ -n "$host_ip" ]]; do
  echo "Generating SSH key on $host_ip..."
  
  ssh "$host_ip" "[ -f ~/.ssh/id_rsa.pub ] || (echo 'Creating SSH key on $host_ip'; ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N '' -q)"
  
  ssh "$host_ip" "[ -f ~/.ssh/id_rsa.pub ]" && echo "Key successfully created on $host_ip" || echo "Failed to create SSH key on $host_ip. Please check connection or permissions."
done < "$WORKERS_FILE"

ALL_KEYS_FILE="all_worker_keys.tmp"
> "$ALL_KEYS_FILE" # Clear file if it exists

while IFS= read -r host_ip || [[ -n "$host_ip" ]]; do
  echo "Collecting public key from $host_ip..."
  
  # Retrieve the public key and append to the temporary file
  ssh "$host_ip" "cat ~/.ssh/id_rsa.pub" >> "$ALL_KEYS_FILE" 2>/dev/null
  if [[ $? -ne 0 ]]; then
    echo "Warning: Could not collect key from $host_ip. SSH key might not have been generated."
  else
    echo "Collected key from $host_ip"
  fi
done < "$WORKERS_FILE"

while IFS= read -r host_ip || [[ -n "$host_ip" ]]; do
  echo "Copying all collected public keys to $host_ip..."

  # Send the file with all public keys to each worker node's authorized_keys
  ssh "$host_ip" "cat >> ~/.ssh/authorized_keys" < "$ALL_KEYS_FILE" && echo "Copied keys to $host_ip"
done < "$WORKERS_FILE"

rm "$ALL_KEYS_FILE"
echo "SSH key setup complete. All workers should now be able to SSH into each other."
