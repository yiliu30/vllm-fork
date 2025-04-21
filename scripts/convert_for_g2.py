import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os


import argparse


weight_factor = (
    torch.finfo(torch.float8_e4m3fnuz).max
    / torch.finfo(torch.float8_e4m3fn).max
)
scale_factor = 1.0 / weight_factor
scale_inv_factor = weight_factor


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".py"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )


def convert_files(input_path, output_path):
    for safetensors_path in glob(f"{input_path}/*.safetensors"):
        tensors = {}
        print(f"processing {safetensors_path}")
        with safe_open(
            safetensors_path, framework="pt", device="cpu"
        ) as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                # tensor = tensor.squeeze(-1)
                if "proj" in k:
                    if k.endswith("weight"):
                        tensor = (tensor.float() * weight_factor).to(
                            torch.float8_e4m3fn
                        )
                    elif k.endswith("weight_scale") or k.endswith(
                        "input_scale"
                    ):
                        tensor = tensor.float() * scale_factor
                    elif k.endswith("weight_scale_inv") or k.endswith(
                        "input_scale_inv"
                    ):
                        # "scale_inv" in deepseek-r1 is actually "scale"
                        tensor = tensor.float() * scale_factor
                    else:
                        raise NotImplementedError(f"Cannot covert {k}")
                else:
                    print(f"skip {k}.")
                k = k.replace("input_scale_inv", "input_scale")
                tensors[k] = tensor
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8 format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="/mnt/disk2/hf_models/DeepSeek-R1",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/mnt/disk2/hf_models/DeepSeek-R1-G2",
        help="Path to the output directory.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path)
