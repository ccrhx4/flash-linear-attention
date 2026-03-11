import torch
import time
import pandas as pd
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

class CacheFlusher:
    def __init__(self, size_mb=128):
        # L2 on L20 is ~50MB-100MB. 128MB-256MB is safe to wipe it.
        self.buffer = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    
    def flush(self):
        self.buffer.zero_() # Forces a write to all cache lines
        torch.cuda.synchronize()

def run_realistic_benchmark(configs):
    results = []
    H_QK, H_V, D_HEAD = 16, 32, 128
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'
    flusher = CacheFlusher(size_mb=256) # Larger than L2 cache

    for conf in configs:
        n_p, p_len, n_d = conf['n_prefill'], conf['avg_prefill_len'], conf['n_decode']
        num_seqs = n_p + n_d
        seq_lens = [p_len] * n_p + [1] * n_d
        total_tokens = sum(seq_lens)
        
        # 1. Setup Tensors
        q = torch.randn(1, total_tokens, H_QK, D_HEAD, dtype=DTYPE, device=DEVICE)
        k = torch.randn(1, total_tokens, H_QK, D_HEAD, dtype=DTYPE, device=DEVICE)
        v = torch.randn(1, total_tokens, H_V, D_HEAD, dtype=DTYPE, device=DEVICE)
        g = torch.randn(1, total_tokens, H_V, dtype=DTYPE, device=DEVICE)
        beta = torch.randn(1, total_tokens, H_V, dtype=DTYPE, device=DEVICE)
        
        # State: [num_seqs, H_V, D_QK, D_V]
        # In real inference, this matrix is read from HBM every time
        initial_state = torch.randn(num_seqs, H_V, D_HEAD, D_HEAD, dtype=torch.float32, device=DEVICE)
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), 
                                  dtype=torch.int32, device=DEVICE)

        # Warmup
        for _ in range(5):
            _ = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu_seqlens, initial_state=initial_state)

        # 2. Realistic Measurement
        iters = 50
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iters):
            flusher.flush() # Force data to be re-read from HBM
            _, _ = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu_seqlens, initial_state=initial_state)
        
        torch.cuda.synchronize()
        # Subtract the known flush time if you want pure op latency, 
        # but usually, the re-fetch IS the real-world latency.
        latency = (time.time() - start) / iters * 1000
        tps = total_tokens / (latency / 1000)
        
        results.append({
            "Scenario": f"{n_p}P+{n_d}D",
            "Total Tokens": total_tokens,
            "Prefill BSxLen": f"{n_p}x{p_len}",
            "Decode Tokens": n_d,
            "Latency (ms)": round(latency, 2),
            "Throughput (tok/s)": round(tps, 0)
        })

    return pd.DataFrame(results)

configs = [
    {"n_prefill": 4, "avg_prefill_len": 2048, "n_decode": 0},
    {"n_prefill": 1, "avg_prefill_len": 4096, "n_decode": 32},
    {"n_prefill": 0, "avg_prefill_len": 0,    "n_decode": 128}
]

print(run_realistic_benchmark(configs).to_string(index=False))