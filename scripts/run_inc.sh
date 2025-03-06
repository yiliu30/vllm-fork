#!/bin/bash
set -x

# Function to load the configuration file
load_config() {
    CONFIG_FILE="devices.conf"
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
    else
        echo "Config file not found!"
        exit 1
    fi
}

# Function to start the Ray worker node
start_worker() {
    BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
    source "$BASH_DIR"/"$WORKER_NODE_SOURCE_FILENAME"
    export QUANT_CONFIG=$INC_MEASURE_CONFIG_FILENAME

    ray start --address="$HEAD_NODE_IP:$RAY_CLUSTER_PORT"

    # check ray status
    sleep 3
    ray status

    # start quant
    sleep 3
    echo "Starting prepare"
    python inc_example_two_nodes.py --mode prepare --smoke
}

# Function to start the Ray head node
start_head() {
    BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
    source "$BASH_DIR"/"$HEAD_NODE_SOURCE_FILENAME"
    export QUANT_CONFIG=$INC_MEASURE_CONFIG_FILENAME

    ray start --head --port $RAY_CLUSTER_PORT
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --node)
            NODE_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if NODE_TYPE was set
if [[ -z "$NODE_TYPE" ]]; then
    echo "Error: --node argument is required. Please specify 'head' or 'worker'."
    exit 1
fi

# Load configuration
load_config

# Run the selected mode
if [[ "$NODE_TYPE" == "head" ]]; then
    echo "Starting Ray head node..."
    start_head
elif [[ "$NODE_TYPE" == "worker" ]]; then
    echo "Starting Ray worker node..."
    start_worker
else
    echo "Invalid node type: $NODE_TYPE. Please specify 'head' or 'worker'."
    exit 1
fi

set +x

# bash run_inc.sh --node head