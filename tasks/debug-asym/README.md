# INC W4A16 Asymmetric Quantization — Not Supported

Asymmetric quantization (`sym=false`) is not supported for INC W4A16 on XPU.

Loading an asymmetric GPTQ model via INC will raise `NotImplementedError` at
model init time. Only symmetric (`sym=true`) models are supported.

## Why

The oneDNN `int4_gemm_w4a16` kernel expects asymmetric zero points as packed u4
nibbles (`dnnl::memory({{num_groups, n}, u4, {n, 1}}`), but the previous INC
code produced `int32 [ngroups, out]` — an 8× size mismatch that causes the
kernel to read garbage.

The test script below reproduces and documents the issue.

## To add asymmetric support later

The qzeros need to be unpacked from GPTQ int32 format, have the GPTQ v1 `+1`
offset applied per-nibble, then repacked into u4 (two nibbles per uint8 byte).
See `test_asym_qzeros.py` for a verified implementation of this repack.

## Test

```bash
cd /home/yiliu7/workspace/vllm
source .venv/bin/activate
python tasks/debug-asym/test_asym_qzeros.py
```

## References

| File | Description |
|------|-------------|
| `vllm/model_executor/layers/quantization/inc.py` | INC quantization — asserts `sym=true` for XPU W4A16 |
| `vllm-xpu-kernels/csrc/xpu/onednn/int4_gemm_w4a16.h` | oneDNN kernel — zp dispatch at lines 67-78, u4 memory at 127-141 |
| `auto-round/auto_round/export/export_to_autogptq/qlinear_triton.py` | Reference GPTQ packing code |
| `tasks/debug-asym/test_asym_qzeros.py` | Reproduction test with proposed fix |
