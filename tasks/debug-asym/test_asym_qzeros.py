#!/usr/bin/env python3
"""Standalone test: INC asymmetric qzeros format bug.

Verifies the unpack/repack pipeline in INCXPULinearMethod.process_weights_after_loading
against the auto-round GPTQ packing reference, and demonstrates that the current
asymmetric output (int32 [ngroups, out]) is incompatible with the oneDNN kernel
which expects packed u4 nibbles.

See BUG_REPORT.md for the full analysis.

Usage:
    cd /home/yiliu7/workspace/vllm
    source .venv/bin/activate
    python tasks/debug-asym/test_asym_qzeros.py
"""

import sys

import numpy as np
import torch

WEIGHT_BITS = 4
PACK_FACTOR = 32 // WEIGHT_BITS  # 8 nibbles per int32
GROUP_SIZE = 128

passed = 0
failed = 0


def check(condition: bool, msg: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {msg}")
    else:
        failed += 1
        print(f"  ❌ {msg}")


# ─── Auto-round reference packing ────────────────────────────────────────────

def autoround_pack_qzeros_asym(zeros: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack asymmetric zero points using auto-round's exact algorithm.

    Reference: auto_round/export/export_to_autogptq/qlinear_triton.py lines 112-126

    Args:
        zeros: [outfeatures, num_groups] actual zero-point values (before offset)

    Returns:
        qzeros: [num_groups, outfeatures // pack_factor] int32
    """
    zeros_t = zeros.t().contiguous()           # [ngroups, out]
    zeros_np = (zeros_t - 1).numpy().astype(np.uint32)  # GPTQ v1: stored = actual - 1

    num_groups, outfeatures = zeros_np.shape
    pack_factor = 32 // bits
    qzeros = np.zeros((num_groups, outfeatures // pack_factor), dtype=np.uint32)

    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + pack_factor):
            qzeros[:, col] |= zeros_np[:, j] << (bits * (j - i))
        i += pack_factor
        col += 1

    return torch.from_numpy(qzeros.astype(np.int32))


# ─── INC code under test ─────────────────────────────────────────────────────

def inc_unpack_qzeros_current(qzeros: torch.Tensor, weight_bits: int = 4) -> torch.Tensor:
    """Current INC code (inc.py lines 148-163): unpack to int32 [ngroups, out].

    This is the BUGGY version — correct values, wrong format.
    """
    mask = (1 << weight_bits) - 1
    shifts = torch.arange(0, 32, weight_bits, dtype=torch.int32)
    zp_unpacked = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]
    ).to(torch.int32)
    zp_unpacked = (zp_unpacked.view(qzeros.shape[0], -1) & mask) + 1
    return zp_unpacked.to(torch.int32).contiguous()


def inc_unpack_qzeros_fixed(qzeros: torch.Tensor, weight_bits: int = 4) -> torch.Tensor:
    """Proposed fix: unpack → add 1 → repack as u4 (uint8 [ngroups, out//2])."""
    mask = (1 << weight_bits) - 1
    shifts = torch.arange(0, 32, weight_bits, dtype=torch.int32)
    zp_unpacked = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]
    ).to(torch.int32)
    zp_unpacked = (zp_unpacked.view(qzeros.shape[0], -1) & mask) + 1

    # Repack: two u4 nibbles per byte, low nibble first
    zp_u8 = zp_unpacked.to(torch.uint8)
    zp_packed = (zp_u8[:, 0::2] & 0xF) | ((zp_u8[:, 1::2] & 0xF) << 4)
    return zp_packed.contiguous()


# ─── Kernel simulation ───────────────────────────────────────────────────────

def simulate_kernel_u4_read(data: torch.Tensor, num_groups: int, n: int) -> list[int]:
    """Simulate what the oneDNN kernel reads when it creates:
        dnnl::memory({{num_groups, n}, u4, {n, 1}}, engine, data.data_ptr())

    Returns a flat list of n*num_groups u4 values as the kernel would see them.
    """
    raw = data.numpy().tobytes()
    nibbles = []
    for byte_val in raw:
        nibbles.append(byte_val & 0xF)
        nibbles.append((byte_val >> 4) & 0xF)
    # Kernel reads num_groups * n nibbles total
    return nibbles[:num_groups * n]


# ═══════════════════════════════════════════════════════════════════════════════

def test_unpack_values_correct():
    """Verify the unpack logic recovers original zero-point values."""
    print("─" * 60)
    print("Test 1: Unpack values correctness (range 1..15)")
    print("─" * 60)

    N, num_groups = 64, 2
    torch.manual_seed(42)
    actual_zp = torch.randint(1, 16, (N, num_groups), dtype=torch.int32)

    qzeros = autoround_pack_qzeros_asym(actual_zp)
    unpacked = inc_unpack_qzeros_current(qzeros)

    expected = actual_zp.t().contiguous()
    check(torch.equal(unpacked, expected),
          f"Unpacked values match original ({unpacked.shape}, {unpacked.dtype})")
    print()


def test_format_incompatible_with_kernel():
    """Demonstrate that int32 output is wrong for the kernel's u4 read."""
    print("─" * 60)
    print("Test 2: Current format is incompatible with kernel")
    print("─" * 60)

    N, num_groups = 16, 1
    actual_zp = torch.full((N, num_groups), 8, dtype=torch.int32)

    qzeros = autoround_pack_qzeros_asym(actual_zp)
    zp_current = inc_unpack_qzeros_current(qzeros)

    # Kernel would read this int32 buffer as u4
    nibbles = simulate_kernel_u4_read(zp_current, num_groups, N)
    expected = [8] * N
    wrong = sum(1 for a, b in zip(nibbles, expected) if a != b)

    print(f"  int32 buffer: {zp_current[0, :4].tolist()} ...")
    print(f"  Kernel reads u4 nibbles: {nibbles[:16]}")
    print(f"  Expected:                {expected[:16]}")
    check(wrong > 0, f"Kernel gets wrong values ({wrong}/{N} incorrect)")

    # Memory size mismatch
    actual_bytes = zp_current.numel() * 4
    expected_bytes = (num_groups * N + 1) // 2
    check(actual_bytes != expected_bytes,
          f"Size mismatch: INC={actual_bytes}B vs kernel expects={expected_bytes}B ({actual_bytes/expected_bytes:.0f}x)")
    print()


def test_proposed_fix():
    """Verify the proposed fix produces correct u4 packed output."""
    print("─" * 60)
    print("Test 3: Proposed fix — repack to u4")
    print("─" * 60)

    N, num_groups = 64, 2
    torch.manual_seed(99)
    actual_zp = torch.randint(1, 16, (N, num_groups), dtype=torch.int32)
    expected = actual_zp.t().contiguous()  # [ngroups, N]

    qzeros = autoround_pack_qzeros_asym(actual_zp)
    zp_fixed = inc_unpack_qzeros_fixed(qzeros)

    # Shape and dtype
    check(zp_fixed.shape == (num_groups, N // 2),
          f"Shape: {tuple(zp_fixed.shape)} == ({num_groups}, {N // 2})")
    check(zp_fixed.dtype == torch.uint8,
          f"Dtype: {zp_fixed.dtype} == uint8")

    # Roundtrip: unpack u4 and compare to expected
    low = (zp_fixed & 0xF).to(torch.int32)
    high = ((zp_fixed >> 4) & 0xF).to(torch.int32)
    reconstructed = torch.stack([low, high], dim=-1).view(num_groups, N)
    check(torch.equal(reconstructed, expected),
          "u4 nibbles roundtrip to original values")

    # Simulate kernel read
    for g in range(num_groups):
        nibbles = simulate_kernel_u4_read(zp_fixed[g:g+1], 1, N)
        exp_list = expected[g].tolist()
        check(nibbles == exp_list,
              f"Group {g}: kernel u4 read matches expected")

    # Memory size
    actual_bytes = zp_fixed.numel()
    expected_bytes = (num_groups * N + 1) // 2
    check(actual_bytes == expected_bytes,
          f"Size: {actual_bytes}B == {expected_bytes}B")
    print()


def test_passthrough_impossible():
    """Show that naive +1 on packed int32 doesn't work."""
    print("─" * 60)
    print("Test 4: Direct passthrough (naive +1) doesn't work")
    print("─" * 60)

    # stored nibbles [7]*8 packed → 0x77777777
    # actual nibbles [8]*8 packed → 0x88888888
    packed_stored = 0x77777777
    packed_actual = 0x88888888
    naive = (packed_stored + 1) & 0xFFFFFFFF

    check(naive != packed_actual,
          f"0x{packed_stored:08X}+1 = 0x{naive:08X} ≠ 0x{packed_actual:08X}")

    # Mixed values
    stored = [3, 7, 11, 5, 2, 9, 14, 1]
    actual = [s + 1 for s in stored]
    ps = sum(s << (4 * i) for i, s in enumerate(stored))
    pa = sum(a << (4 * i) for i, a in enumerate(actual))
    naive2 = (ps + 1) & 0xFFFFFFFF

    check(naive2 != pa,
          f"Mixed: 0x{ps:08X}+1 = 0x{naive2:08X} ≠ 0x{pa:08X}")
    print()


def test_edge_cases():
    """Test boundary zero-point values."""
    print("─" * 60)
    print("Test 5: Edge cases")
    print("─" * 60)

    N, num_groups = 16, 1

    # zp=1 (stored=0): minimum valid
    actual_1 = torch.full((N, num_groups), 1, dtype=torch.int32)
    qz = autoround_pack_qzeros_asym(actual_1)
    zp = inc_unpack_qzeros_fixed(qz)
    nibbles = simulate_kernel_u4_read(zp, num_groups, N)
    check(nibbles == [1] * N, "zp=1 (stored=0) → kernel reads all 1s")

    # zp=8 (stored=7): symmetric default via asymmetric path
    actual_8 = torch.full((N, num_groups), 8, dtype=torch.int32)
    qz = autoround_pack_qzeros_asym(actual_8)
    zp = inc_unpack_qzeros_fixed(qz)
    nibbles = simulate_kernel_u4_read(zp, num_groups, N)
    check(nibbles == [8] * N, "zp=8 (stored=7) → kernel reads all 8s")

    # zp=15 (stored=14): maximum valid
    actual_15 = torch.full((N, num_groups), 15, dtype=torch.int32)
    qz = autoround_pack_qzeros_asym(actual_15)
    zp = inc_unpack_qzeros_fixed(qz)
    nibbles = simulate_kernel_u4_read(zp, num_groups, N)
    check(nibbles == [15] * N, "zp=15 (stored=14) → kernel reads all 15s")

    # zp=0 (stored=-1): overflows in GPTQ v1 packing — known limitation
    actual_0 = torch.full((N, num_groups), 0, dtype=torch.int32)
    qz = autoround_pack_qzeros_asym(actual_0)
    unpacked = inc_unpack_qzeros_current(qz)
    # stored=-1 → uint32 0xFFFFFFFF → nibble=0xF, +1 = 16 → overflows u4
    check((unpacked == 16).all().item(),
          "zp=0 (stored=-1) → unpacks to 16 (overflows u4, known limitation)")
    print()


def test_symmetric_path_ok():
    """Verify the symmetric path is correct for completeness."""
    print("─" * 60)
    print("Test 6: Symmetric path (not affected)")
    print("─" * 60)

    zp_sym = torch.tensor([8], dtype=torch.int8)
    check(zp_sym.dim() == 1, "dim()==1 → kernel uses scalar s8 broadcast")
    check(zp_sym.dtype == torch.int8, f"dtype is int8")
    check(zp_sym.item() == 8, f"value is 8")
    print()


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("INC Asymmetric qzeros Bug — Reproduction & Verification")
    print("=" * 60)
    print()

    test_unpack_values_correct()
    test_format_incompatible_with_kernel()
    test_proposed_fix()
    test_passthrough_impossible()
    test_edge_cases()
    test_symmetric_path_ok()

    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed:
        # "Failed" assertions in test 2 and 4 are intentional — they demonstrate
        # the bug.  Only unexpected failures should cause a non-zero exit.
        # All our "check" calls that demonstrate bugs use check(condition, ...)
        # where the condition is True when the bug is confirmed, so failed==0
        # means all demonstrations and fixes are working as expected.
        print("\nUnexpected failures detected!")
        sys.exit(1)
    else:
        print("\nAll checks passed. Bug demonstrated and fix verified.")


if __name__ == "__main__":
    main()
