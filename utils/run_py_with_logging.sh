#!/bin/bash

# run_py_with_logging.sh
#
# This script executes a Python script and redirects its output to log files.
# It captures the exit code and generates a summary log, which can be used
# in CI/CD pipelines to report the status of the execution.

# Configuration
LOG_DIR="$1"
SCRIPT_PATH="$2"
PIPELINE_NAME="$3"
BUILD_ID="$4"
OUTPUT_LOG="$LOG_DIR/output.log"
SUMMARY_LOG="$LOG_DIR/summary.log"
STATUS_FILE="$LOG_DIR/status.txt"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "Starting execution of $SCRIPT_PATH..."

# Run the Python script
# Redirect stdout and stderr to the output log
python "$SCRIPT_PATH" > "$OUTPUT_LOG" 2>&1

# Capture exit code
EXIT_CODE=$?

# Create the Summary
{
    echo "=== Pipeline Run Summary ==="
    echo "Pipeline: $PIPELINE_NAME"
    echo "Build ID: $BUILD_ID"
    echo "Timestamp: $(date)"
    echo "Exit Code: $EXIT_CODE"
    echo "---------------------------"
    echo "Last 20 lines of log:"
    tail -n 20 "$OUTPUT_LOG"
} > "$SUMMARY_LOG"

# If it failed, create the status file for the YAML to detect
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED" > "$STATUS_FILE"
    echo "Script failed with exit code $EXIT_CODE. Summary generated."
    exit 0 # We exit 0 here so the pipeline continues to the "Check Status" step
else
    echo "Script completed successfully."
fi
