#!/bin/bash
# This script launches vLLM in the background, redirecting output to a log file.

# Set the model directory/path and desired port
MODEL="Qwen/Qwen2.5-3B-Instruct"

# Create a safe filename by replacing slashes with underscores
SAFE_MODEL_NAME=$(echo $MODEL | tr '/' '_')

# Launch vLLM in background using nohup and redirect both stdout and stderr to a log file.
# nohup poetry run vllm serve $MODEL --tensor-parallel-size 4 > vllm_$SAFE_MODEL_NAME.log 2>&1 &
nohup poetry run vllm serve $MODEL > vllm_$SAFE_MODEL_NAME.log 2>&1 &

# Print a message indicating that vLLM is running.
echo "vLLM has been launched in the background with the $MODEL model. Check vllm_$SAFE_MODEL_NAME.log for output."
