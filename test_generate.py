"""
NOTE: This is for testing only.
"""

import argparse
import json
from typing import AsyncGenerator
import asyncio

from async_llm_engine import AsyncLLMEngine
from sampling_params import SamplingParams
from utils import random_uuid, AsyncEngineArgs


async def generate(request_dict):
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return ret


if __name__ == "__main__":

    engine_args = AsyncEngineArgs(model="mock")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    payload = {
            "prompt": "this is a prompt",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 10,
            "ignore_eos": True,
            "stream": True,
        }

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.gather(
        generate(payload.copy()),
        generate(payload.copy()),
    ))
    print(result)