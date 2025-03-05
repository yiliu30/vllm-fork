#! /bin/bash
set -x

# Load the config file
CONFIG_FILE="devices.conf"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Config file not found!"
    exit 1
fi

# reset the environment and start the ray cluster
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/worker_node_source.sh
export QUANT_CONFIG=$INC_MEASURE_CONFIG_FILENAME


ray start --address="$HEAD_NODE_IP:$RAY_CLUSTER_PORT"