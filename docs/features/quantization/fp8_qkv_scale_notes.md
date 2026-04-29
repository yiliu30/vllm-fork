# FP8 Q/K/V Scale Notes

This note explains the saved `q_scale`, `k_scale`, and `v_scale` values used by `llm-compressor` and `compressed-tensors` for attention FP8.

## Example checkpoint

Reference checkpoint:
`llm-compressor/examples/quantization_attention/Qwen3-8B-attention-fp8`

This checkpoint uses `strategy: tensor`, so each layer stores one scalar `q_scale`, one scalar `k_scale`, and one scalar `v_scale`.

## On-disk values

Across 36 layers:

| scale | mean | min | max |
| --- | ---: | ---: | ---: |
| `q_scale` | `0.03815036` | `0.02172852` | `0.06176758` |
| `k_scale` | `0.07578532` | `0.03564453` | `0.52343750` |
| `v_scale` | `0.03905307` | `0.000766754` | `0.16308594` |

Layer 0 is a useful sanity check:

- `q_scale = 0.03393555`
- `k_scale = 0.52343750`
- `v_scale = 0.000766754`

## How the scales are collected

- `compressed-tensors` creates the attention qparams in `compressed_tensors/quantization/lifecycle/initialize.py`.
- `llm-compressor` attaches `q`, `k`, and `v` observers plus calibration hooks in `llmcompressor/modifiers/quantization/quantization/mixin.py`.
- During calibration, `query` goes through the hooked attention path, while `key` and `value` go through the hooked KV-cache path.
- For `strategy: tensor`, attention states with shape `[B, H, S, D]` are flattened to `(B*S, 1, H*D)` before min/max is taken.
- With symmetric FP8 E4M3, the scale is `absmax / 448`.

## What the scale means

These are normal quantization scales, not inverse scales.

```text
x_q = round(x / scale)
x ~= x_q * scale
```

That is the same convention used by `compressed-tensors` in `quantization/lifecycle/forward_helpers.py`.

## Multiply or divide in attention

Multiply.

If you keep `Q` and `K` in quantized form, the float score should be:

```text
QK ~= (Q_q @ K_q^T) * q_scale * k_scale
```

Then apply the usual attention factor:

```text
QK ~= (Q_q @ K_q^T) * q_scale * k_scale / sqrt(d_head)
```

If `V` also stays quantized through the attention path, its output needs `* v_scale`.

Division by `scale` only happens during quantization. The matmul result should not divide by `q_scale` or `k_scale`.

## Important caveat

This recipe uses `memoryless_minmax`.

That means the saved activation scales are overwritten on each calibration step. The final `q_scale`, `k_scale`, and `v_scale` on disk are the last observed calibration values on the saving rank, not a running aggregate over the full calibration dataset.
