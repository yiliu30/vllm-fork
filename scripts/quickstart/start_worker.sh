#!/bin/bash
source ./set_worker_node.sh
ray start --address='10.239.128.244:8850'

sleep 10

ray status