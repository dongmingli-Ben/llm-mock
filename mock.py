"""This module contains mock functions to simulate the compute time of LLM workloads"""

from scheduler import SchedulerTask
from typing import List
import asyncio
from config import GPU_COMPUTE_POWER, GPU_MEMORY_BANDWIDTH, MODEL_SIZE

async def mock_execute_model_async(tasks: List[SchedulerTask]):
    """Mock function to simulate the compute time of LLM workloads"""
    seq_len, new_token_len = [], []
    for task in tasks:
        seq_len.append(len(task.token_ids))
        new_token_len.append(len(task.token_ids) - task.first_new_token_idx)
    bs = len(tasks)
    tensor_length = max(seq_len) if len(seq_len) else 0
    compute_length = max(new_token_len) if (len(new_token_len)) else 0  # only this part needs to compute KV
    compute_workload = bs * compute_length * tensor_length
    model_size = MODEL_SIZE  # GB
    compute_time = compute_workload / GPU_COMPUTE_POWER
    memory_time = model_size / GPU_MEMORY_BANDWIDTH
    print(f"Mocking compute time: {compute_time + memory_time:.4f} seconds")
    await asyncio.sleep(compute_time + memory_time)
    # await asyncio.sleep(0.1)