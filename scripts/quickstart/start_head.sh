#!/bin/bash

source ./set_head_node.sh
ray start --head --node-ip-address=10.239.128.244 --port=8850  --num-cpus 160
