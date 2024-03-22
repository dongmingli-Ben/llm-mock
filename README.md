

## Benchmarking

```bash
python benchmark_serving.py --dataset small-1k.json --num-prompts 50
```

### Plain Inference

Result:

```
Successful requests: 47
Benchmark duration: 134.275680 s
Total input tokens: 7830
Total generated tokens: 7642
Request throughput: 0.35 requests/s
Input token throughput: 58.31 tokens/s
Output token throughput: 56.91 tokens/s
Mean TTFT: 71036.62 ms
Median TTFT: 76935.37 ms
P99 TTFT: 126153.82 ms
Mean TPOT: 667.68 ms
Median TPOT: 498.57 ms
P99 TPOT: 1272.80 ms
```

### Cache Inference

500 cache entry result:

```
Successful requests: 47
Benchmark duration: 128.044960 s
Total input tokens: 7830
Total generated tokens: 7642
Request throughput: 0.37 requests/s
Input token throughput: 61.15 tokens/s
Output token throughput: 59.68 tokens/s
Mean TTFT: 71340.69 ms
Median TTFT: 80767.82 ms
P99 TTFT: 122566.54 ms
Mean TPOT: 679.42 ms
Median TPOT: 703.25 ms
P99 TPOT: 1217.45 ms
```

1k cache entry result:

```
Successful requests: 47
Benchmark duration: 126.786682 s
Total input tokens: 7830
Total generated tokens: 7642
Request throughput: 0.37 requests/s
Input token throughput: 61.76 tokens/s
Output token throughput: 60.27 tokens/s
Mean TTFT: 71280.88 ms
Median TTFT: 75978.79 ms
P99 TTFT: 121699.89 ms
Mean TPOT: 674.62 ms
Median TPOT: 591.63 ms
P99 TPOT: 1212.05 ms
```

5k:

```
Successful requests: 47
Benchmark duration: 120.244815 s
Total input tokens: 7830
Total generated tokens: 7642
Request throughput: 0.39 requests/s
Input token throughput: 65.12 tokens/s
Output token throughput: 63.55 tokens/s
Mean TTFT: 58228.22 ms
Median TTFT: 62479.08 ms
P99 TTFT: 112241.96 ms
Mean TPOT: 567.64 ms
Median TPOT: 448.02 ms
P99 TPOT: 1125.68 ms
```