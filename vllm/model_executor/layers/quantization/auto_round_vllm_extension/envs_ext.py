# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# FILE: main.py

import os
from typing import Any, Callable


from vllm.logger import init_logger

# FILE: extra_envs.py

logger = init_logger(__name__)

# Define extra environment variables
extra_environment_variables: dict[str, Callable[[], Any]] = {
    # Example: Define an extra environment variable for enabling experimental features
    "VLLM_AR_MXFP8_DISABLE_INPUT_QDQ": lambda: os.getenv("VLLM_AR_MXFP8_DISABLE_INPUT_QDQ", "0")
    in ("1", "true", "True"),
    "VLLM_USE_STATIC_MOE_HPU": lambda: os.getenv("VLLM_USE_STATIC_MOE_HPU", "0") in ("1", "true", "True"),
}
from vllm.envs import environment_variables

# Merge the environment variables
all_environment_variables = {**environment_variables, **extra_environment_variables}

# Add the extra environment variables to vllm.envs
import vllm.envs as envs

for name, value_fn in extra_environment_variables.items():
    setattr(envs, name, value_fn())

logger.warning_once("Added extra environment variables: " f"{list(extra_environment_variables.keys())}")
