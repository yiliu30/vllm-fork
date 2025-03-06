import os
import subprocess
import time
import sys
import configparser


import os
import subprocess
import time
import sys
import configparser

# Constants
RAY_CLUSTER_PORT = "RAY_CLUSTER_PORT"
HEAD_NODE_IP = "HEAD_NODE_IP"
HEAD_NODE_IFNAME = "HEAD_NODE_IFNAME"
HEAD_NODE_SOURCE_FILENAME = "HEAD_NODE_SOURCE_FILENAME"
WORKER_NODE_IP = "WORKER_NODE_IP"
WORKER_NODE_IFNAME = "WORKER_NODE_IFNAME"
WORKER_NODE_SOURCE_FILENAME = "WORKER_NODE_SOURCE_FILENAME"
INC_MEASURE_CONFIG_FILENAME = "INC_MEASURE_CONFIG_FILENAME"
INC_QUANT_CONFIG_FILENAME = "INC_QUANT_CONFIG_FILENAME"

config_file = "devices.conf"


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.sections():
        print("Config file not found or is empty!")
        sys.exit(1)
    for section in config.sections():
        for key, value in config.items(section):
            os.environ[key] = value
            print(f"Set {key} to {value}")


def is_head(args):
    return args.node == "head"


def is_quant(args):
    return args.mode == "quant"


def is_prepare(args):
    return args.mode == "prepare"


def main(args):
    # Enable debug mode
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDEBUG"] = "1"

    # Load the config file
    load_config(config_file)

    # Reset the environment and start the ray cluster
    bash_dir = os.path.dirname(os.path.abspath(__file__))
    if is_head(args):
        source_file = os.environ.get(HEAD_NODE_SOURCE_FILENAME)
    else:
        source_file = os.environ.get(WORKER_NODE_SOURCE_FILENAME)

    if source_file:
        node_source_path = os.path.join(bash_dir, source_file)
        subprocess.run(
            ["bash", "-c", f"source {node_source_path}"], check=True, shell=True
        )

    if is_quant(args):
        INC_QUANT_CONFIG = os.environ.get(INC_QUANT_CONFIG_FILENAME, "")
    elif is_prepare(args):
        INC_QUANT_CONFIG = os.environ.get(INC_MEASURE_CONFIG_FILENAME, "")
    else:
        INC_QUANT_CONFIG = None

    # export QUANT_CONFIG
    if INC_QUANT_CONFIG:
        subprocess.run(
            ["bash", "-c", f"export QUANT_CONFIG={INC_QUANT_CONFIG}"],
            check=True,
            shell=True,
        )

    # start ray
    if is_head(args):
        node_ip = os.environ.get(HEAD_NODE_IP)
    else:
        node_ip = os.environ.get(WORKER_NODE_IP)
    ray_cluster_port = os.environ.get(RAY_CLUSTER_PORT)
    assert (
        node_ip and ray_cluster_port
    ), "Node IP and Ray Cluster Port must be set!"

    subprocess.run(
        ["ray", "start", f"--address={node_ip}:{ray_cluster_port}"],
        check=True,
        shell=True,
    )

    # Check ray status
    time.sleep(3)
    subprocess.run(["ray", "status"], check=True)

    # Start quant
    time.sleep(3)
    if is_prepare(args):
        cmd_mode = "prepare"
    elif is_quant(args):
        cmd_mode = "quant"
    else:
        cmd_mode = None
    print(f"Starting inc_example_two_nodes.py with mode {cmd_mode}")
    subprocess.run(
        ["python", "inc_example_two_nodes.py", "--mode", cmd_mode, "--smoke"],
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--node",
        choices=["head", "worker"],
        help="The node type.",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["quant", "prepare"],
        help="The mode.",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    parser.add_argument(
        "--fp8_kvcache", action="store_true", help="Using FP8 KV cache."
    )
    args = parser.parse_args()
    main(args)

    # python run_inc.py --node head --mode prepare --smoke
    # python run_inc.py --node worker --mode quant --smoke
