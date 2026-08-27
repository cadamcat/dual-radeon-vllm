"""Minimal fixtures so these two files can run standalone in the container."""
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture()
def dist_init():
    from vllm.distributed import (cleanup_dist_env_and_memory,
                                  init_distributed_environment,
                                  initialize_model_parallel)
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{temp_file}",
        local_rank=0, backend="gloo")
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()
