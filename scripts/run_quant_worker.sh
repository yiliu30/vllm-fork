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
export QUANT_CONFIG=$INC_QUANT_CONFIG_FILENAME

ray start --address="$HEAD_NODE_IP:$RAY_CLUSTER_PORT"

# check ray status
sleep 3
ray status


# start quant
sleep 3
echo "Starting quant"
python inc_example_two_nodes.py --mode quant --smoke

set +x