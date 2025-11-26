# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from typing_extensions import override

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


from .gpu_profiler import WorkerProfiler
class XPUTorchProfilerWrapper(WorkerProfiler):
    def __init__(self, worker_name: str, local_rank: int) -> None:
        super().__init__()

        self.local_rank = local_rank
        torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
        if local_rank in (None, 0):
            logger.info(
                "Torch profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
            profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
            with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
            with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir, worker_name=worker_name, use_gzip=True
            ),
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

        rank = self.local_rank
        profiler_dir = envs.VLLM_TORCH_PROFILER_DIR
        profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
        sort_key = "self_xpu_time_total"
        table = self.profiler.key_averages().table(sort_by=sort_key)

        with open(profiler_out_file, "w") as f:
            print(table, file=f)

        # only print profiler results on rank 0
        if rank == 0:
            print(table)

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)

