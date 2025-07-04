import torch
import vllm.envs as envs
from vllm.scalar_type import scalar_types



from torchao.prototype.mx_formats.mx_tensor import to_mx, to_dtype, ScaleCalculationMode

def run_mxfp4_emulations(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
):
    group_size = 32
    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mx(data_hp=x,
                             elem_dtype="fp4_e2m1",
                             block_size=group_size,
                             scaling_mode=ScaleCalculationMode.RCEIL,
                             pack_fp6=False)

    # dequantize input
    x_dq = to_dtype(data_lp=x_q, 
                    scale_e8m0=input_scale,
                    elem_dtype="fp4_e2m1",
                    block_size=group_size,
                    target_dtype=x.dtype,
                    use_fp4_custom_triton_dequant_kernel=False,
                    pack_fp6=False)

    # dequantize weight
    w_dq = to_dtype(data_lp=weight, 
                    scale_e8m0=weight_scale,
                    elem_dtype="fp4_e2m1",
                    block_size=group_size,
                    target_dtype=torch.bfloat16,
                    use_fp4_custom_triton_dequant_kernel=False,
                    pack_fp6=False)

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out