#!/bin/bash
set -x

# Function to load the configuration file
load_config() {
    CONFIG_FILE="devices.conf"
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        echo "Config file sourced successfully"
    else
        echo "Config file not found!"
        exit 1
    fi
}

# Function to start the Ray worker node
start_worker() {
    BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
    source "$BASH_DIR"/"$WORKER_NODE_SOURCE_FILENAME"
    
    # Check if quant mode is selected
    if [[ "$MODE" == "quant" ]]; then
        export QUANT_CONFIG=$INC_quant_CONFIG_FILENAME
    else
        export QUANT_CONFIG=$INC_MEASURE_CONFIG_FILENAME
    fi

    ray start --address="$HEAD_NODE_IP:$RAY_CLUSTER_PORT"

    # Check Ray status
    sleep 3
    ray status

    # Start quant
    sleep 3
    echo "Starting prepare"
    python inc_example_two_nodes.py --mode prepare --smoke
}

# Function to start the Ray head node
start_head() {
    BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
    source "$BASH_DIR"/"$HEAD_NODE_SOURCE_FILENAME"
    
    # Check if quant mode is selected
    if [[ "$MODE" == "quant" ]]; then
        export QUANT_CONFIG=$INC_quant_CONFIG_FILENAME
    else
        export QUANT_CONFIG=$INC_MEASURE_CONFIG_FILENAME
    fi

    ray start --head --port $RAY_CLUSTER_PORT
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --node)
            NODE_TYPE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if NODE_TYPE and MODE were set
if [[ -z "$NODE_TYPE" ]]; then
    echo "Error: --node argument is required. Please specify 'head' or 'worker'."
    exit 1
fi

if [[ -z "$MODE" ]]; then
    echo "Error: --mode argument is required. Please specify 'quant' or 'measure'."
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

# Example to run:
# bash run_inc.sh --node head --mode quant
# bash run_inc.sh --node worker --mode measure
