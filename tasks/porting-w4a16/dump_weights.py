"""Exhaustive comparison: dump the FULL fused qweight from vLLM vs manual concat.
Writes tensors to files so we can compare across processes."""

import os
import torch
from torch.nn.parameter import Parameter

# Patch process_weights_after_loading to save pre-repack weights
import vllm.model_executor.layers.quantization.inc as inc_mod

OrigProcess = inc_mod.INCXPULinearMethod.process_weights_after_loading
_save_count = [0]
SAVE_DIR = "/home/yiliu7/workspace/vllm/tasks/porting-w4a16/dumped_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

def patched_process(self, layer):
    _save_count[0] += 1
    idx = _save_count[0]
    # Save the first 4 layers (qkv, o, gate_up, down of layer 0)
    if idx <= 4:
        torch.save(layer.qweight.data.cpu(), f"{SAVE_DIR}/qweight_{idx}.pt")
        torch.save(layer.scales.data.cpu(), f"{SAVE_DIR}/scales_{idx}.pt")
        torch.save(layer.qzeros.data.cpu(), f"{SAVE_DIR}/qzeros_{idx}.pt")
        print(f"[SAVED] Layer {idx}: qweight={layer.qweight.shape}, "
              f"scales={layer.scales.shape}", flush=True)
    OrigProcess(self, layer)

inc_mod.INCXPULinearMethod.process_weights_after_loading = patched_process

from vllm import LLM, SamplingParams
llm = LLM(
    model='Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound',
    block_size=64, enforce_eager=True, max_model_len=256,
    gpu_memory_utilization=0.5,
)
out = llm.generate(['The capital of France is'],
                    SamplingParams(max_tokens=5, temperature=0))
print('OUTPUT:', repr(out[0].outputs[0].text))
