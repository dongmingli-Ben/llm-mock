

## Benchmarking

```bash
python benchmark_serving.py --dataset small-1k.json --num-prompts 50
```

### Plain Inference

Result:

```
Successful requests: 47
Benchmark duration: 109.354170 s
Total input tokens: 7830
Total generated tokens: 5956
Request throughput: 0.43 requests/s
Input token throughput: 71.60 tokens/s
Output token throughput: 54.47 tokens/s
Mean TTFT: 61358.14 ms
Median TTFT: 71804.48 ms
P99 TTFT: 105076.36 ms
Mean TPOT: 642.22 ms
Median TPOT: 747.60 ms
P99 TPOT: 967.10 ms
```

### Cache Inference

1k cache entry result:

```
Successful requests: 47
Benchmark duration: 104.008613 s
Total input tokens: 7830
Total generated tokens: 5668
Request throughput: 0.45 requests/s
Input token throughput: 75.28 tokens/s
Output token throughput: 54.50 tokens/s
Mean TTFT: 60639.36 ms
Median TTFT: 66616.12 ms
P99 TTFT: 99913.21 ms
Mean TPOT: 647.29 ms
Median TPOT: 695.18 ms
P99 TPOT: 944.88 ms
```