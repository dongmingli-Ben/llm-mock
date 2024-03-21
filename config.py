"""Global configuration for mocking"""

import random

MIN_DECODE_LENGTH = 50
MAX_DECODE_LENGTH = 150

def random_decode_length():
    return random.randint(MIN_DECODE_LENGTH, MAX_DECODE_LENGTH)

BATCH_SIZE = 10

GPU_COMPUTE_POWER = 312 * 100 * 15  # TFLOPS * coefficient
GPU_MEMORY_BANDWIDTH = 2000  # GB/s
MODEL_SIZE = 120  # GB

USE_CACHE = True
MAX_CACHE_TOKEN_SIZE = 1000