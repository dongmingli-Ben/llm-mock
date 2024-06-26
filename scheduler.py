"""Scheduler for requests"""

from dataclasses import dataclass
from typing import List, Optional
from config import BATCH_SIZE, USE_CACHE, MAX_CACHE_TOKEN_SIZE, decode_length
from sampling_params import SamplingParams
from utils import CompletionOutput
from cache import NoCache, PrefixTreeCache, PromptCacheBase
from copy import deepcopy

@dataclass
class SchedulerTask:
    """Task for the scheduler"""
    
    request_id: str = None
    prompt: str = None
    prompt_token_ids: List[int] = None
    sampling_params: Optional[SamplingParams] = None
    arrival_time: float = 0.0
    token_ids: List[int] = None
    first_new_token_idx: int = 0

    output: CompletionOutput = None
    """This field is meant to make the step function easier."""
    decode_length: int = 0
    """The length of the decoded sequence."""


class Scheduler:

    def __init__(self, batch_size: int = BATCH_SIZE, cache_prompts: List[List[int]] = None):
        self.task_pool: List[SchedulerTask] = []
        self.batch_size = batch_size
        self.cache: PromptCacheBase = PrefixTreeCache(MAX_CACHE_TOKEN_SIZE) \
            if USE_CACHE else NoCache()
        if cache_prompts:
            self.cache.populate(cache_prompts)

    def add_request(self, 
                    request_id: str, 
                    prompt: str, 
                    prompt_token_ids: List[int], 
                    sampling_params: Optional[SamplingParams] = None,
                    arrival_time: float = 0.0):
        """Add a request to the scheduler"""
        first_new_token_idx = self.cache.query_first_different_token_idx(prompt_token_ids)
        task = SchedulerTask(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=arrival_time,
            token_ids=prompt_token_ids.copy(),
            first_new_token_idx=first_new_token_idx,
            decode_length=decode_length(len(prompt_token_ids)),
            output=CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                finish_reason=None
            )
        )
        print(f"NEW REQUEST: first new token index: {first_new_token_idx}, decode length: {task.decode_length}")
        self.task_pool.append(task)

    def schedule(self) -> List[SchedulerTask]:
        """Schedule the next batch of requests"""
        return self.task_pool[:self.batch_size]
    
    def update_request_decode(self, 
                              request_id: str,
                              token_id: int,
                              finish: bool) -> Optional[CompletionOutput]:
        """Update the tasks according to the newly decoded token.
        
        Args:
            request_id: The ID of the request.
            token_id: The token ID decoded.
            finish: Whether the sequence is finished. When the sequence is finished, 
                the token_id does not matter.

        Returns:
            The copy of the CompletionOutput.
        """
        for i, task in enumerate(self.task_pool):
            if task.request_id == request_id:
                if finish:
                    self.task_pool.pop(i)
                    task.output.finish_reason = "length"
                    return task.output
                task.token_ids.append(token_id)
                task.first_new_token_idx = len(task.token_ids) - 1
                task.output.token_ids.append(token_id)
                task.output.text += f' #{token_id}'
                return deepcopy(task.output)