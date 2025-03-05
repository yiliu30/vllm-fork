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
source "$BASH_DIR"/head_node_source.sh
export QUANT_CONFIG=$INC_QUANT_CONFIG_FILENAME
ray start --head --port $RAY_CLUSTER_PORT