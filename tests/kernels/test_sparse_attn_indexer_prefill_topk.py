# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import _prefill_topk_funnel_dense
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_top_k_per_row_prefill_ragged_local_indices_and_padding() -> None:
    """Validate ragged prefill semantics: local indices and -1 padding."""
    torch.set_default_device("cuda:0")

    num_rows = 4
    top_k = 3
    max_cols = 12
    row_starts = torch.tensor([2, 0, 4, 7], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([10, 2, 5, 7], dtype=torch.int32, device="cuda")

    logits = torch.full(
        (num_rows, max_cols),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    # Row 0 valid range [2, 10): length 8, k=3 should pick top-3 by value.
    logits[0, 2:10] = torch.tensor(
        [1.0, 9.0, 5.0, 7.0, 8.0, 6.0, 4.0, 3.0],
        device="cuda",
    )
    # Row 1 valid range [0, 2): length 2, one padded output expected.
    logits[1, 0:2] = torch.tensor([4.0, 3.0], device="cuda")
    # Row 2 valid range [4, 5): length 1, two padded outputs expected.
    logits[2, 4:5] = torch.tensor([8.0], device="cuda")
    # Row 3 valid range [7, 7): empty row, all outputs should be -1.

    # Inject large values outside valid ranges; they must be ignored.
    logits[0, 0] = 1000.0
    logits[1, 11] = 1000.0
    logits[2, 0] = 1000.0

    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )

    for i in range(num_rows):
        start = int(row_starts[i])
        end = int(row_ends[i])
        valid_len = max(0, end - start)
        k_i = min(top_k, valid_len)

        if k_i == 0:
            assert torch.all(indices[i] == -1)
            continue

        # Valid outputs are local indices into [0, valid_len).
        local = indices[i, :k_i]
        assert torch.all(local >= 0)
        assert torch.all(local < valid_len)

        # No duplicate picks among selected local indices.
        assert torch.unique(local).numel() == k_i

        # Padded outputs must be -1.
        if k_i < top_k:
            assert torch.all(indices[i, k_i:] == -1)

        # Selected values must match torch.topk on the row-local valid slice
        # (order is not required here).
        global_idx = (local + start).to(torch.int64)
        selected_vals = logits[i, global_idx].sort(descending=True)[0]
        ref_vals = torch.topk(logits[i, start:end], k_i, dim=-1, sorted=True)[0]
        assert torch.allclose(selected_vals, ref_vals, rtol=1e-5, atol=1e-5)

        # Large out-of-range logits must never be selected.
        assert torch.all(global_idx >= start)
        assert torch.all(global_idx < end)

    # Specific sanity check: row 0 should not include column 0 despite huge value.
    row0_global = indices[0, :top_k].to(torch.int64) + int(row_starts[0])
    assert 0 not in row0_global.cpu().tolist()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_prefill_topk_funnel_dense_matches_native_semantics() -> None:
    """Experimental funnel_dense prefill path should match native semantics."""
    try:
        funnel_spec = importlib.util.find_spec("funnel_topk.funnel")
    except ModuleNotFoundError:
        funnel_spec = None

    if funnel_spec is None:
        pytest.skip("funnel_topk is not importable in this environment")

    torch.set_default_device("cuda:0")
    num_rows = 32
    num_cols = 513
    top_k = 300

    for seed in range(6):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        row_starts = torch.randint(
            0,
            num_cols + 1,
            (num_rows,),
            dtype=torch.int32,
            device="cuda",
            generator=gen,
        )
        row_ends = torch.randint(
            0,
            num_cols + 1,
            (num_rows,),
            dtype=torch.int32,
            device="cuda",
            generator=gen,
        )
        row_ends = torch.maximum(row_ends, row_starts)

        logits = torch.randn(
            num_rows,
            num_cols,
            dtype=torch.float32,
            device="cuda",
            generator=gen,
        )

        for row in range(num_rows):
            start = int(row_starts[row])
            end = int(row_ends[row])
            if start > 0:
                logits[row, :start] = 1e6
            if end < num_cols:
                logits[row, end:] = 1e6

        native_indices = torch.empty(
            (num_rows, top_k),
            dtype=torch.int32,
            device="cuda",
        )
        dense_indices = torch.empty(
            (num_rows, top_k),
            dtype=torch.int32,
            device="cuda",
        )

        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            native_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        _prefill_topk_funnel_dense(
            logits,
            row_starts,
            row_ends,
            dense_indices,
            top_k,
        )

        for row in range(num_rows):
            start = int(row_starts[row])
            end = int(row_ends[row])
            valid_len = max(0, end - start)
            k_i = min(top_k, valid_len)

            if k_i == 0:
                assert torch.all(native_indices[row] == -1)
                assert torch.all(dense_indices[row] == -1)
                continue

            native_local = native_indices[row, :k_i]
            dense_local = dense_indices[row, :k_i]

            assert torch.all(dense_local >= 0)
            assert torch.all(dense_local < valid_len)

            if k_i < top_k:
                assert torch.all(native_indices[row, k_i:] == -1)
                assert torch.all(dense_indices[row, k_i:] == -1)

            native_global = native_local.to(torch.int64) + start
            dense_global = dense_local.to(torch.int64) + start

            assert torch.all(dense_global >= start)
            assert torch.all(dense_global < end)

            native_vals = torch.sort(logits[row, native_global], descending=True)[0]
            dense_vals = torch.sort(logits[row, dense_global], descending=True)[0]
            assert torch.allclose(native_vals, dense_vals, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_prefill_topk_funnel_dense_import_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """funnel_dense path should fail clearly when funnel_topk import fails."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "funnel_topk.funnel":
            raise ModuleNotFoundError("mock missing funnel_topk")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    logits = torch.zeros((1, 4), dtype=torch.float32)
    row_starts = torch.tensor([0], dtype=torch.int32)
    row_ends = torch.tensor([4], dtype=torch.int32)
    topk_indices = torch.empty((1, 2), dtype=torch.int32)

    with pytest.raises(
        RuntimeError,
        match=(
            "VLLM_SPARSE_INDEXER_PREFILL_TOPK_BACKEND=funnel_dense "
            "requires funnel_topk to be importable"
        ),
    ):
        _prefill_topk_funnel_dense(
            logits,
            row_starts,
            row_ends,
            topk_indices,
            topk_tokens=2,
        )
