import uuid
from dataclasses import dataclass
from typing import Optional, List


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_paddings: int = 256
    max_logprobs: int = 5  # OpenAI default value
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    tokenizer_pool_type: str = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    lora_extra_vocab_size: int = 256
    lora_dtype = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    ray_workers_use_nsight: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        finish_reason: The reason why the sequence is finished.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request. Each output is a complete
            sequence.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished})")