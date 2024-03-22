"""Global configuration for mocking"""

MIN_DECODE_LENGTH = 50
MAX_DECODE_LENGTH = 150
PROMPT_DECODE_THRESHOLD = 500

def decode_length(prompt_length: int):
    return MIN_DECODE_LENGTH + \
        int(min(prompt_length,  PROMPT_DECODE_THRESHOLD) * \
            (MAX_DECODE_LENGTH - MIN_DECODE_LENGTH) / PROMPT_DECODE_THRESHOLD)

BATCH_SIZE = 10

GPU_COMPUTE_POWER = 312 * 100 * 15  # TFLOPS * coefficient
GPU_MEMORY_BANDWIDTH = 2000  # GB/s
MODEL_SIZE = 120  # GB

USE_CACHE = True
MAX_CACHE_TOKEN_SIZE = 500