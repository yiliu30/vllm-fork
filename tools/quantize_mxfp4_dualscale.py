# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert a BF16 Hugging Face checkpoint to the vLLM dual-scale MXFP4 format.

The converter uses RTN-style MXFP4 quantization: each 512-value K block gets an
FP32 coarse scale, and the normalized values get the regular E2M1/E8M0 MXFP4
quantization with 32-value fine blocks.  It is intended for quick kernel and
model integration tests; it does not run AutoRound's calibration/optimization.

Example:

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python tools/quantize_mxfp4_dualscale.py \\
        --model /dev/shm/.tmp_yi/Qwen/Qwen3-8B \\
        --output /dev/shm/.tmp_yi/Qwen/Qwen3-8B-MXFP4-dualscale \\
        --device cuda:0
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp,
)


COARSE_K = 512
FINE_K = 32
DEFAULT_COARSE_MAX = 6.0
DEFAULT_REFERENCE_CONFIG = Path(
    "/dev/shm/.tmp_yi/workspace/auto-round/"
    "Qwen3-8B-MXFP4_RCEIL/Qwen3-8B-mxfp-w4g32"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/dev/shm/.tmp_yi/Qwen/Qwen3-8B"),
        help="Input BF16 Hugging Face model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/dev/shm/.tmp_yi/Qwen/Qwen3-8B-MXFP4-dualscale"),
        help="Output directory for the dual-scale checkpoint.",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=DEFAULT_REFERENCE_CONFIG,
        help="Existing MXFP4 directory whose quantization settings are reused.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for quantization; default: cuda:0.",
    )
    parser.add_argument(
        "--coarse-max",
        type=float,
        default=DEFAULT_COARSE_MAX,
        help=(
            "Numerator used for coarse_scale=amax/coarse_max; "
            f"default: {DEFAULT_COARSE_MAX}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def _quantize_weight(
    weight: torch.Tensor, device: torch.device, coarse_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an [out_features, in_features] weight with two scale levels."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got shape {weight.shape}")
    if weight.shape[1] % COARSE_K != 0:
        raise ValueError(
            f"Input dimension {weight.shape[1]} is not divisible by {COARSE_K}"
        )
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Expected floating-point weights, got {weight.dtype}")

    if coarse_max <= 0:
        raise ValueError(f"coarse_max must be positive, got {coarse_max}")

    weight_device = weight.to(device=device, dtype=torch.float32)
    out_features, in_features = weight_device.shape
    coarse = weight_device.reshape(
        out_features, in_features // COARSE_K, COARSE_K
    ).abs().amax(dim=-1)
    coarse = coarse / coarse_max
    coarse = torch.where(coarse == 0, torch.ones_like(coarse), coarse)
    normalized = (
        weight_device.reshape(out_features, in_features // COARSE_K, COARSE_K)
        / coarse[..., None]
    ).reshape(out_features, in_features)

    # Keep the input precision expected by the vLLM Triton quantizer.  The
    # quantizer computes its block maxima in FP32 internally.
    quant_input = normalized.to(
        torch.bfloat16 if weight.dtype == torch.bfloat16 else torch.float16
    )
    packed, fine, _ = downcast_to_mxfp(
        quant_input,
        axis=1,
        BLOCK_OUT_DIM=128,
        BLOCK_QUANT_DIM=COARSE_K,
    )
    return packed.cpu(), fine.cpu(), coarse.cpu()


def _is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return whether *name* is a dense model-layer Linear weight."""
    return (
        name.startswith("model.layers.")
        and name.endswith(".weight")
        and tensor.ndim == 2
    )


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with index_path.open() as file:
        index = json.load(file)
    return index["weight_map"]


def _write_config(
    model_dir: Path, output_dir: Path, reference_dir: Path
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    with config_path.open() as file:
        config = json.load(file)

    reference_path = reference_dir / "quantization_config.json"
    if reference_path.exists():
        with reference_path.open() as file:
            quant_config = json.load(file)
    else:
        quant_config = {
            "bits": 4,
            "act_bits": 4,
            "data_type": "mx_fp4e2m1",
            "act_data_type": "mx_fp4e2m1",
            "group_size": FINE_K,
            "act_group_size": FINE_K,
            "sym": True,
            "act_sym": True,
            "act_dynamic": True,
            "enable_quanted_input": False,
            "quant_method": "auto-round",
            "packing_format": "auto_round:llm_compressor",
            "block_name_to_quantize": "model.layers",
        }

    # The reference config is retained for compatibility with INC's existing
    # AutoRound dispatch.  The dual-scale flag selects the new vLLM kernel.
    quant_config["dual_scale"] = True
    quant_config["bits"] = 4
    quant_config["data_type"] = "mx_fp4e2m1"
    quant_config["act_data_type"] = "mx_fp4e2m1"
    quant_config["group_size"] = FINE_K
    quant_config["act_bits"] = 4
    quant_config["act_group_size"] = FINE_K
    quant_config["block_name_to_quantize"] = "model.layers"
    quant_config["packing_format"] = "auto_round:llm_compressor"
    quant_config["quant_method"] = "auto-round"
    config["quantization_config"] = quant_config

    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    return quant_config


def _copy_non_weight_files(model_dir: Path, output_dir: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
    }
    for source in model_dir.iterdir():
        if source.name in excluded or source.name.endswith(".safetensors"):
            continue
        destination = output_dir / source.name
        if source.is_file():
            shutil.copy2(source, destination)


def convert(model_dir: Path, output_dir: Path, reference_dir: Path,
            device: str, coarse_max: float, overwrite: bool) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Input model directory does not exist: {model_dir}")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Triton MXFP4 quantizer")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    weight_map = _load_weight_map(model_dir)
    input_shards = sorted(set(weight_map.values()))
    output_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_index, shard_name in enumerate(input_shards, start=1):
        source_path = model_dir / shard_name
        output_tensors: dict[str, torch.Tensor] = {}
        output_name = f"model-{shard_index:05d}-of-{len(input_shards):05d}.safetensors"
        output_path = output_dir / output_name
        print(f"[{shard_index}/{len(input_shards)}] {source_path.name}")

        with safe_open(str(source_path), framework="pt", device="cpu") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                if _is_linear_weight(name, tensor):
                    packed, fine, coarse = _quantize_weight(
                        tensor, device_obj, coarse_max
                    )
                    quantized_tensors = {
                        f"{name}_packed": packed,
                        f"{name}_scale": fine,
                        f"{name}_coarse_scale": coarse,
                    }
                    print(
                        f"  quantized {name}: {tuple(tensor.shape)} -> "
                        f"packed={tuple(packed.shape)} coarse={tuple(coarse.shape)}"
                    )
                else:
                    quantized_tensors = {name: tensor}

                for output_key, output_tensor in quantized_tensors.items():
                    output_tensors[output_key] = output_tensor
                    output_weight_map[output_key] = output_name
                    total_size += output_tensor.nbytes
                del tensor
                if device_obj.type == "cuda":
                    torch.cuda.empty_cache()

        save_file(output_tensors, str(output_path), metadata={"format": "pt"})
        del output_tensors

    _copy_non_weight_files(model_dir, output_dir)
    _write_config(model_dir, output_dir, reference_dir)
    index = {
        "metadata": {
            "format": "safetensors",
            "total_shards": len(input_shards),
            "total_size": total_size,
        },
        "weight_map": output_weight_map,
    }
    with (output_dir / "model.safetensors.index.json").open("w") as file:
        json.dump(index, file, indent=2)
        file.write("\n")
    print(f"Wrote dual-scale checkpoint to {output_dir}")


def main() -> None:
    args = _parse_args()
    convert(
        model_dir=args.model,
        output_dir=args.output,
        reference_dir=args.reference_config,
        device=args.device,
        coarse_max=args.coarse_max,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
