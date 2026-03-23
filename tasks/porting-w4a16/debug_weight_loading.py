#!/usr/bin/env python3
"""
Debug script: Compare raw checkpoint weights vs vLLM-loaded weights
for Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound.

This script checks whether vLLM's QKV fusion (shard concatenation) corrupts
the packed int32 qweight values when merging q_proj, k_proj, v_proj into
qkv_proj.

Model info:
  - quant_method: auto-round
  - packing_format: auto_round:auto_awq  (AWQ-style: [input, output/pack])
  - bits: 4, group_size: 128, sym: true
  - pack_factor: 8 (8 x int4 packed into 1 x int32)

Checkpoint shapes for layer 0 q_proj:
  - qweight: [896, 112] int32  (896 input, 896/8=112 output_packed)
  - scales:  [7, 896] fp16    (7 groups of 128, 896 output)
  - qzeros:  [7, 112] int32   (7 groups, 896/8=112 output_packed)

vLLM fuses q_proj/k_proj/v_proj into qkv_proj with
output_partition_sizes=[896, 128, 128] (total=1152).
  - Fused qweight: [896, 144] int32  (1152/8=144 output_packed)
  - Fused scales:  [7, 1152] fp16
  - Fused qzeros:  [7, 144] int32   (1152/8=144)

Key question: Does concatenation along the packed output dimension preserve
the int32 bit patterns? Answer: YES, because the shard sizes (896, 128, 128)
are all divisible by pack_factor=8, so each shard maps to whole int32 elements.
The packed int32 values are simply placed side-by-side, not bit-interleaved.

Usage:
    cd /home/yiliu7/workspace/vllm
    source .venv/bin/activate
    python tasks/porting-w4a16/debug_weight_loading.py
"""

import gc
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound"
LAYER_PREFIX = "model.layers.0.self_attn"
PROJ_NAMES = ["q_proj", "k_proj", "v_proj"]
WEIGHT_SUFFIXES = ["qweight", "scales", "qzeros"]
PACK_FACTOR = 8  # 32 // 4
NUM_DISPLAY = 8  # how many values to print in previews

# Qwen2-0.5B attention dimensions
HEAD_SIZE = 64
NUM_Q_HEADS = 14  # total_num_heads
NUM_KV_HEADS = 2  # total_num_kv_heads
Q_OUTPUT = NUM_Q_HEADS * HEAD_SIZE  # 896
K_OUTPUT = NUM_KV_HEADS * HEAD_SIZE  # 128
V_OUTPUT = NUM_KV_HEADS * HEAD_SIZE  # 128
TOTAL_OUTPUT = Q_OUTPUT + K_OUTPUT + V_OUTPUT  # 1152

SEPARATOR = "=" * 78


def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ===========================================================================
# Part 1: Load raw checkpoint weights via safetensors
# ===========================================================================
def load_raw_checkpoint():
    """Load individual q/k/v weight tensors directly from safetensors files."""
    print_section("PART 1: Loading raw checkpoint via safetensors")

    from safetensors.torch import load_file

    # Find the safetensors file(s) - may be cached by huggingface_hub
    from huggingface_hub import snapshot_download

    print(f"Downloading/locating model: {MODEL_ID}")
    model_dir = snapshot_download(MODEL_ID)
    print(f"Model directory: {model_dir}")

    # Load all safetensors files
    st_files = sorted(Path(model_dir).glob("*.safetensors"))
    print(f"Found {len(st_files)} safetensors file(s): {[f.name for f in st_files]}")

    raw_weights = {}
    for proj in PROJ_NAMES:
        for suffix in WEIGHT_SUFFIXES:
            key = f"{LAYER_PREFIX}.{proj}.{suffix}"
            for sf in st_files:
                tensors = load_file(str(sf))
                if key in tensors:
                    raw_weights[key] = tensors[key]
                    break
            if key not in raw_weights:
                print(f"  WARNING: {key} not found in any safetensors file!")

    # Print raw checkpoint info
    print(f"\nRaw checkpoint weights for {LAYER_PREFIX}:")
    for proj in PROJ_NAMES:
        print(f"\n  --- {proj} ---")
        for suffix in WEIGHT_SUFFIXES:
            key = f"{LAYER_PREFIX}.{proj}.{suffix}"
            if key in raw_weights:
                t = raw_weights[key]
                print(f"  {suffix:>10s}: shape={list(t.shape)}, dtype={t.dtype}")
                if suffix == "qweight":
                    flat = t.flatten()
                    print(
                        f"             first {NUM_DISPLAY} int32 values: "
                        f"{flat[:NUM_DISPLAY].tolist()}"
                    )
                    print(
                        f"             first {NUM_DISPLAY} hex: "
                        f"{[hex(v) for v in flat[:NUM_DISPLAY].tolist()]}"
                    )
                elif suffix == "scales":
                    print(
                        f"             [0, :4] = {t[0, :4].tolist()}"
                    )
                elif suffix == "qzeros":
                    flat = t.flatten()
                    print(
                        f"             first {NUM_DISPLAY} int32 values: "
                        f"{flat[:NUM_DISPLAY].tolist()}"
                    )

    return raw_weights


# ===========================================================================
# Part 2: Load model via vLLM and capture weights BEFORE process_weights
# ===========================================================================
def load_via_vllm():
    """
    Load model via vLLM's LLM class with a monkey-patched
    process_weights_after_loading to capture the fused weights
    BEFORE the marlin repacking transforms them.
    """
    print_section("PART 2: Loading model via vLLM (with weight capture hook)")

    # We'll capture the pre-processed (but post-fusion) weights here
    captured_weights = {}

    # Monkey-patch process_weights_after_loading to capture weights
    import vllm.model_executor.model_loader.utils as loader_utils

    _original_process = loader_utils.process_weights_after_loading

    def _patched_process(model, model_config, target_device):
        """Capture fused weights before process_weights_after_loading."""
        print("\n  [HOOK] Capturing fused weights BEFORE process_weights_after_loading")

        # Strategy 1: Walk named_parameters for layer 0 QKV
        print("  [HOOK] Scanning model.named_parameters()...")
        for name, param in model.named_parameters():
            if "layers.0" in name and "self_attn" in name and "qkv_proj" in name:
                # Extract the suffix after "qkv_proj."
                short_name = name.split("self_attn.")[-1]
                captured_weights[short_name] = param.data.clone().cpu()
                print(
                    f"  [HOOK] Captured {name}: "
                    f"shape={list(param.shape)}, dtype={param.dtype}"
                )

        # Strategy 2: Walk named_modules to find the qkv_proj module directly
        if not captured_weights:
            print("  [HOOK] Scanning model.named_modules()...")
            for name, module in model.named_modules():
                if "layers.0.self_attn.qkv_proj" in name:
                    for attr in ["qweight", "scales", "qzeros"]:
                        if hasattr(module, attr):
                            param = getattr(module, attr)
                            key = f"qkv_proj.{attr}"
                            captured_weights[key] = param.data.clone().cpu()
                            print(
                                f"  [HOOK] Captured {name}.{attr}: "
                                f"shape={list(param.shape)}, "
                                f"dtype={param.dtype}"
                            )
                    break

        # Now call the original
        _original_process(model, model_config, target_device)
        print("  [HOOK] process_weights_after_loading completed")

    loader_utils.process_weights_after_loading = _patched_process

    try:
        from vllm import LLM

        print(f"\nInitializing vLLM LLM with model: {MODEL_ID}")
        print("  Options: enforce_eager=True, max_model_len=4096, block_size=64")

        llm = LLM(
            model=MODEL_ID,
            enforce_eager=True,
            max_model_len=4096,
            block_size=64,
            gpu_memory_utilization=0.5,
        )

        print("\n  vLLM model loaded successfully.")
    finally:
        # Restore original
        loader_utils.process_weights_after_loading = _original_process

    return captured_weights, llm


# ===========================================================================
# Part 3: Compare raw checkpoint vs vLLM fused weights
# ===========================================================================
def compare_weights(raw_weights: dict, captured_weights: dict):
    """
    Compare raw per-proj weights from checkpoint against fused QKV weights
    loaded by vLLM.

    The fused qweight should be the concatenation of q/k/v qweights along
    the output (packed) dimension:
      q_proj.qweight [896, 112] | k_proj.qweight [896, 16] | v_proj.qweight [896, 16]
      => qkv_proj.qweight [896, 144]
    """
    print_section("PART 3: Comparing raw checkpoint vs vLLM fused weights")

    if not captured_weights:
        print("  ERROR: No weights were captured from vLLM. "
              "The hook may not have triggered.")
        print("  Available captured keys:", list(captured_weights.keys()))
        return False

    print(f"  Captured weight keys: {list(captured_weights.keys())}")

    # Find the qweight in captured weights
    qweight_key = None
    for k in captured_weights:
        if "qweight" in k:
            qweight_key = k
            break

    if qweight_key is None:
        print("  ERROR: No qweight found in captured weights!")
        return False

    fused_qweight = captured_weights[qweight_key]
    print(f"\n  Fused qweight (from vLLM): shape={list(fused_qweight.shape)}, "
          f"dtype={fused_qweight.dtype}")

    # --- Compare qweight ---
    print_section("PART 3a: qweight comparison")

    # Determine shard boundaries on the packed output dimension (dim=1)
    # For AWQ packing: qweight is [input, output/pack_factor]
    # output_partition_sizes = [896, 128, 128]
    # packed: [896/8, 128/8, 128/8] = [112, 16, 16] => total 144
    q_packed = Q_OUTPUT // PACK_FACTOR  # 112
    k_packed = K_OUTPUT // PACK_FACTOR  # 16
    v_packed = V_OUTPUT // PACK_FACTOR  # 16

    shard_info = {
        "q_proj": (0, q_packed),              # [0:112]
        "k_proj": (q_packed, k_packed),        # [112:128]
        "v_proj": (q_packed + k_packed, v_packed),  # [128:144]
    }

    all_match = True
    for proj, (offset, size) in shard_info.items():
        raw_key = f"{LAYER_PREFIX}.{proj}.qweight"
        if raw_key not in raw_weights:
            print(f"  WARNING: {raw_key} not found in raw weights, skipping")
            continue

        raw_qw = raw_weights[raw_key]
        fused_shard = fused_qweight[:, offset:offset + size]

        print(f"\n  {proj}.qweight:")
        print(f"    Raw checkpoint:  shape={list(raw_qw.shape)}")
        print(f"    Fused shard [{offset}:{offset+size}]: shape={list(fused_shard.shape)}")

        if raw_qw.shape != fused_shard.shape:
            print(f"    *** SHAPE MISMATCH! raw={list(raw_qw.shape)} "
                  f"vs fused={list(fused_shard.shape)}")
            all_match = False
            continue

        # Element-wise comparison
        match = torch.equal(raw_qw, fused_shard)
        num_diff = (raw_qw != fused_shard).sum().item()
        total = raw_qw.numel()

        print(f"    Exact match: {match}")
        if not match:
            print(f"    Mismatched elements: {num_diff}/{total} "
                  f"({100*num_diff/total:.2f}%)")
            # Show first few differences
            diff_mask = raw_qw != fused_shard
            diff_indices = diff_mask.nonzero()[:5]
            for idx in diff_indices:
                i, j = idx.tolist()
                print(f"      [{i},{j}] raw=0x{raw_qw[i,j].item():08x} "
                      f"vs fused=0x{fused_shard[i,j].item():08x}")
            all_match = False
        else:
            # Show a few matching values to confirm
            print(f"    Sample values (first row, first {min(4, size)} packed):")
            for j in range(min(4, size)):
                val = raw_qw[0, j].item()
                print(f"      [{0},{j}] = 0x{val:08x} ({val})")

    # --- Compare scales ---
    print_section("PART 3b: scales comparison")

    scales_key = None
    for k in captured_weights:
        if "scales" in k:
            scales_key = k
            break

    if scales_key:
        fused_scales = captured_weights[scales_key]
        print(f"  Fused scales: shape={list(fused_scales.shape)}, "
              f"dtype={fused_scales.dtype}")

        scales_shard_info = {
            "q_proj": (0, Q_OUTPUT),
            "k_proj": (Q_OUTPUT, K_OUTPUT),
            "v_proj": (Q_OUTPUT + K_OUTPUT, V_OUTPUT),
        }

        for proj, (offset, size) in scales_shard_info.items():
            raw_key = f"{LAYER_PREFIX}.{proj}.scales"
            if raw_key not in raw_weights:
                continue
            raw_sc = raw_weights[raw_key]
            fused_shard = fused_scales[:, offset:offset + size]

            match = torch.equal(raw_sc, fused_shard)
            print(f"\n  {proj}.scales: raw={list(raw_sc.shape)}, "
                  f"fused_shard=[{offset}:{offset+size}] -> exact_match={match}")
            if not match:
                num_diff = (raw_sc != fused_shard).sum().item()
                print(f"    Mismatched elements: {num_diff}/{raw_sc.numel()}")
                # Check if it's just fp precision
                max_diff = (raw_sc.float() - fused_shard.float()).abs().max().item()
                print(f"    Max abs diff: {max_diff}")
                all_match = False
    else:
        print("  No scales captured")

    # --- Compare qzeros ---
    print_section("PART 3c: qzeros comparison")

    qzeros_key = None
    for k in captured_weights:
        if "qzeros" in k:
            qzeros_key = k
            break

    if qzeros_key:
        fused_qzeros = captured_weights[qzeros_key]
        print(f"  Fused qzeros: shape={list(fused_qzeros.shape)}, "
              f"dtype={fused_qzeros.dtype}")

        qzeros_shard_info = {
            "q_proj": (0, q_packed),
            "k_proj": (q_packed, k_packed),
            "v_proj": (q_packed + k_packed, v_packed),
        }

        for proj, (offset, size) in qzeros_shard_info.items():
            raw_key = f"{LAYER_PREFIX}.{proj}.qzeros"
            if raw_key not in raw_weights:
                continue
            raw_qz = raw_weights[raw_key]
            fused_shard = fused_qzeros[:, offset:offset + size]

            match = torch.equal(raw_qz, fused_shard)
            print(f"\n  {proj}.qzeros: raw={list(raw_qz.shape)}, "
                  f"fused_shard=[{offset}:{offset+size}] -> exact_match={match}")
            if not match:
                num_diff = (raw_qz != fused_shard).sum().item()
                print(f"    Mismatched elements: {num_diff}/{raw_qz.numel()}")
                diff_mask = raw_qz != fused_shard
                diff_indices = diff_mask.nonzero()[:5]
                for idx in diff_indices:
                    i, j = idx.tolist()
                    print(f"      [{i},{j}] raw=0x{raw_qz[i,j].item():08x} "
                          f"vs fused=0x{fused_shard[i,j].item():08x}")
                all_match = False
    else:
        print("  No qzeros captured")

    return all_match


# ===========================================================================
# Part 4: Verify packing integrity via manual unpack-repack test
# ===========================================================================
def verify_packing_integrity(raw_weights: dict):
    """
    Manually simulate the QKV fusion and verify that concatenating packed
    int32 shards along the output dimension preserves the unpacked int4 values.
    """
    print_section("PART 4: Manual packing integrity verification")

    q_qw = raw_weights.get(f"{LAYER_PREFIX}.q_proj.qweight")
    k_qw = raw_weights.get(f"{LAYER_PREFIX}.k_proj.qweight")
    v_qw = raw_weights.get(f"{LAYER_PREFIX}.v_proj.qweight")

    if q_qw is None or k_qw is None or v_qw is None:
        print("  ERROR: Missing raw qweight tensors")
        return

    print(f"  q_proj.qweight: {list(q_qw.shape)}")
    print(f"  k_proj.qweight: {list(k_qw.shape)}")
    print(f"  v_proj.qweight: {list(v_qw.shape)}")

    # Concatenate along output (packed) dimension
    fused = torch.cat([q_qw, k_qw, v_qw], dim=1)
    print(f"  Concatenated: {list(fused.shape)}")

    # Now unpack both the individual and fused tensors
    def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        """Unpack AWQ-style int4: 8 x 4-bit values in each int32, along dim=1."""
        in_size = packed.shape[0]
        out_packed = packed.shape[1]
        out_size = out_packed * PACK_FACTOR
        unpacked = torch.zeros((in_size, out_size), dtype=torch.int32)
        mask = 0xF
        for i in range(PACK_FACTOR):
            unpacked[:, i::PACK_FACTOR] = (packed >> (4 * i)) & mask
        return unpacked

    q_unpacked = unpack_int4(q_qw)
    k_unpacked = unpack_int4(k_qw)
    v_unpacked = unpack_int4(v_qw)
    expected_unpacked = torch.cat([q_unpacked, k_unpacked, v_unpacked], dim=1)

    fused_unpacked = unpack_int4(fused)

    match = torch.equal(expected_unpacked, fused_unpacked)
    print(f"\n  Unpacked match after fused-cat vs individual-cat: {match}")

    if not match:
        num_diff = (expected_unpacked != fused_unpacked).sum().item()
        total = expected_unpacked.numel()
        print(f"  Mismatched: {num_diff}/{total} elements")

        # Show where mismatches occur
        diff_mask = expected_unpacked != fused_unpacked
        diff_indices = diff_mask.nonzero()[:10]
        for idx in diff_indices:
            i, j = idx.tolist()
            print(f"    [{i},{j}] expected={expected_unpacked[i,j].item()} "
                  f"got={fused_unpacked[i,j].item()}")
    else:
        print("  ✓ Concatenating packed int32 shards preserves all int4 values")
        print(f"    (Verified {expected_unpacked.numel()} unpacked int4 values)")

    # Additional check: verify shard sizes are pack_factor-aligned
    print(f"\n  Shard alignment check (must be divisible by pack_factor={PACK_FACTOR}):")
    for name, size in [("q_proj", Q_OUTPUT), ("k_proj", K_OUTPUT),
                       ("v_proj", V_OUTPUT)]:
        aligned = size % PACK_FACTOR == 0
        print(f"    {name} output_size={size}: "
              f"{'✓ aligned' if aligned else '✗ NOT ALIGNED (CORRUPTION RISK!)'}")


# ===========================================================================
# Part 5: Summary of quant method routing
# ===========================================================================
def print_routing_info():
    """Show which quantization method vLLM selects for this model."""
    print_section("PART 5: Quantization method routing analysis")

    print(f"  Model: {MODEL_ID}")
    print(f"  quant_method: auto-round → overridden to 'inc' by INCConfig")
    print(f"  packing_format: auto_round:auto_awq")
    print(f"  Since 'awq' is in packing_format, INCConfig routes to:")
    print(f"    → apply_awq_quant_layer()")
    print(f"    → AWQMarlinLinearMethod (if marlin-compatible on CUDA)")
    print()
    print(f"  AWQMarlinLinearMethod creates:")
    print(f"    qweight: PackedvLLMParameter(packed_dim=1, output_dim=1)")
    print(f"             shape=[input, output/pack_factor]")
    print(f"    qzeros:  PackedvLLMParameter(packed_dim=1, output_dim=1)")
    print(f"             shape=[num_groups, output/pack_factor]")
    print(f"    scales:  GroupQuantScaleParameter()")
    print(f"             shape=[num_groups, output]")
    print()
    print(f"  Since packed_dim == output_dim for qweight and qzeros:")
    print(f"    → QKV shard offsets/sizes ARE adjusted for packing")
    print(f"    → shard_size = output_size / pack_factor")
    print(f"    → shard_offset = output_offset / pack_factor")
    print(f"  This works correctly because all shard sizes are multiples of 8:")
    print(f"    q_proj: {Q_OUTPUT}/8 = {Q_OUTPUT//PACK_FACTOR} packed columns")
    print(f"    k_proj: {K_OUTPUT}/8 = {K_OUTPUT//PACK_FACTOR} packed columns")
    print(f"    v_proj: {V_OUTPUT}/8 = {V_OUTPUT//PACK_FACTOR} packed columns")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(SEPARATOR)
    print("  Debug Weight Loading: Packed Int32 Integrity Check")
    print(f"  Model: {MODEL_ID}")
    print(SEPARATOR)

    # Part 5 first (no loading needed)
    print_routing_info()

    # Part 1: Raw checkpoint
    raw_weights = load_raw_checkpoint()

    # Part 4: Manual integrity check (no vLLM needed)
    verify_packing_integrity(raw_weights)

    # Part 2: vLLM loading
    print("\n" + SEPARATOR)
    print("  Now loading via vLLM to capture actual fused weights...")
    print(SEPARATOR)
    captured_weights, llm = load_via_vllm()

    # Part 3: Compare
    all_match = compare_weights(raw_weights, captured_weights)

    # Final summary
    print_section("FINAL SUMMARY")
    if all_match:
        print("  ✓ ALL CHECKS PASSED")
        print("    The QKV shard concatenation preserves packed int32 values correctly.")
        print("    No corruption detected in the weight loading pipeline.")
    else:
        print("  ✗ SOME CHECKS FAILED")
        print("    The packed int32 values may be corrupted during QKV fusion.")
        print("    See details above for specific mismatches.")

    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
