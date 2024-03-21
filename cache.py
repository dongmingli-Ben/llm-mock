"""A prefix tree-based cache for prompts."""

from typing import List, Dict
from transformers import PreTrainedTokenizer
import json
from dataclasses import dataclass, field
import heapq


@dataclass
class TrieNode:

    path_count: int = 0
    """Number of paths that pass through this node."""
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    parent: "TrieNode" = None
    """Use to make the pruning easier."""
    token_id: int = None
    """The token id that leads to this node."""

    def __del__(self):
        for child in self.children.values():
            del child

    def __lt__(self, other: "TrieNode"):
        return self.token_id < other.token_id  # favor smaller token id


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: List[int]):
        node = self.root
        for token_id in token_ids:
            node.path_count += 1
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
                node.children[token_id].parent = node
                node.children[token_id].token_id = token_id
            node = node.children[token_id]
        node.path_count += 1

    def prune(self, max_edges: int):
        pq = [(-self.root.path_count, self.root)]
        max_edges += 1  # root does not count
        while len(pq) and max_edges:
            neg_count, node = heapq.heappop(pq)
            max_edges -= 1
            for child in node.children.values():
                heapq.heappush(pq, (-child.path_count, child))
        for neg_count, node in pq:
            del node.parent.children[node.token_id]
            del node

    def query_first_different_token_idx(self, token_ids: List[int]) -> int:
        node = self.root
        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                return i
            node = node.children[token_id]
        return len(token_ids)


class PromptCacheBase:

    def __init__(self, max_edges: int = None):
        pass

    def populate(self, prompts: List[List[int]]):
        pass

    def query_first_different_token_idx(self, token_ids: List[int]) -> int:
        pass


class NoCache(PromptCacheBase):

    def query_first_different_token_idx(self, token_ids: List[int]) -> int:
        return 0
    

class PrefixTreeCache(PromptCacheBase):
    
    def __init__(self, max_edges: int):
        self.trie = Trie()
        self.max_edges = max_edges

    def populate(self, prompts: List[List[int]]):
        for prompt in prompts:
            self.trie.insert(prompt)
        self.trie.prune(self.max_edges)
        print(f"CACHE: Prefix tree cache populated with {len(prompts)} prompts.")

    def query_first_different_token_idx(self, token_ids: List[int]) -> int:
        return self.trie.query_first_different_token_idx(token_ids)
    

def encode_prompts_from_file(path: str, tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    """Encode prompts from a file."""
    with open(path, "r") as f:
        data = json.load(f)
        # take the first turn of each conversation as prompts
        prompts = [d['conversations'][0]['value'] for d in data if len(d['conversations']) > 0]
    return tokenizer(prompts).input_ids