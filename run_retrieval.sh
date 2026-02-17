#!/bin/bash
# Background execution tool for retrieval scripts
# Usage: ./run_retrieval.sh <python_script> [output_subdir]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [output_subdir]"
    echo "Example: $0 crires_retrieval_chips_starB.py"
    echo "         $0 crires_retrieval_chips_starB.py CD-35_2722/2022-12-31/starB/whole_chips/7mol_free"
    exit 1
fi

SCRIPT_NAME="$1"
OUTPUT_SUBDIR="${2:-}"

# Get absolute path of script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
OUTPUT_BASE="${SCRIPT_DIR}/output/retrievals"

# Check if Python script exists
if [ ! -f "${SRC_DIR}/${SCRIPT_NAME}" ]; then
    echo "Error: Script not found ${SRC_DIR}/${SCRIPT_NAME}"
    exit 1
fi

# If output_subdir is provided, use it; otherwise try to extract from Python script
if [ -z "$OUTPUT_SUBDIR" ]; then
    # Try to extract output_subdir from Python script (simple pattern matching)
    OUTPUT_SUBDIR=$(grep -oP "output_subdir=['\"]([^'\"]+)['\"]" "${SRC_DIR}/${SCRIPT_NAME}" | head -1 | sed "s/output_subdir=['\"]//;s/['\"]//" || echo "")
fi

# Determine log file path
if [ -n "$OUTPUT_SUBDIR" ]; then
    LOG_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run.log"
else
    # If output_subdir not found, use default log directory
    LOG_DIR="${SCRIPT_DIR}/output/logs"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.py}_$(date +%Y%m%d_%H%M%S).log"
fi

echo "=========================================="
echo "Running script: ${SCRIPT_NAME}"
echo "Output directory: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Activate conda environment and run script (background execution)
cd "${SRC_DIR}" || exit 1

# Use nohup to run in background, redirect all output to log file
# Use python -u to disable output buffering so all print statements are captured immediately
nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate crires+ && python -u ${SCRIPT_NAME}" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process started, PID: ${PID}"
echo "Log file: ${LOG_FILE}"
echo "View log: tail -f ${LOG_FILE}"
echo "Stop process: kill ${PID}"
