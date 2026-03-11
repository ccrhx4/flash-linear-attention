import torch
import time
import pandas as pd
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)

class CacheFlusher:
    def __init__(self, size_mb=256):
        self.buffer = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    
    def flush(self):
        self.buffer.zero_()
        torch.cuda.synchronize()

def mixed_mode_dispatch(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    a,
    b,
    dt_bias,
    cu_seqlens,
    initial_state,
    ssm_state_indices=None,
):
    """
    Logic: 
    1. If all seq_lens > 1 -> Chunk Path (Prefill)
    2. If all seq_lens == 1 -> Recurrent Path (Decode)
    3. If Mixed -> Split and process (or default to Chunk if supported)
    """
    # Calculate individual sequence lengths from cu_seqlens
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    is_pure_decode = bool(torch.all(seq_lens == 1).item())

    if is_pure_decode:
        # Optimized path for BS=0 Prefill (Pure Decode)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens, 
            initial_state=initial_state, 
            ssm_state_indices=ssm_state_indices,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        # Standard path for Prefill or Mixed workloads
        return chunk_gated_delta_rule(
            q, k, v, g, beta, 
            cu_seqlens=cu_seqlens, 
            initial_state=initial_state, 
            output_final_state=True
        )

def run_mixed_benchmark(configs):
    H_QK, H_V, D_HEAD = 16, 32, 128
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'
    flusher = CacheFlusher()
    results = []

    for conf in configs:
        n_p, p_len, n_d = conf['n_prefill'], conf['avg_prefill_len'], conf['n_decode']
        
        # 1. Setup VarLen indexing
        seq_lens = ([p_len] * n_p) + ([1] * n_d)
        total_tokens = sum(seq_lens)
        num_seqs = n_p + n_d
        is_pure_decode_case = n_p == 0 and n_d > 0
        
        if total_tokens == 0: continue

        # 2. Tensors
        q = torch.randn(1, total_tokens, H_QK, D_HEAD, dtype=DTYPE, device=DEVICE)
        k = torch.randn(1, total_tokens, H_QK, D_HEAD, dtype=DTYPE, device=DEVICE)
        v = torch.randn(1, total_tokens, H_V, D_HEAD, dtype=DTYPE, device=DEVICE)
        g = torch.randn(1, total_tokens, H_V, dtype=DTYPE, device=DEVICE)
        beta = torch.rand(1, total_tokens, H_V, dtype=DTYPE, device=DEVICE).sigmoid()
        A_log = torch.randn(H_V, dtype=DTYPE, device=DEVICE)
        dt_bias = torch.randn(H_V, dtype=DTYPE, device=DEVICE)
        a = torch.randn(total_tokens, H_V, dtype=DTYPE, device=DEVICE)
        b = torch.randn(total_tokens, H_V, dtype=DTYPE, device=DEVICE)
        
        if is_pure_decode_case:
            total_entries = total_tokens * 2
            initial_state = torch.randn(
                total_entries,
                H_V,
                D_HEAD,
                D_HEAD,
                dtype=torch.float32,
                device=DEVICE,
            )
            ssm_state_indices = torch.randperm(
                total_entries, dtype=torch.int32, device=DEVICE
            )[:total_tokens]
            cu_seqlens = torch.arange(
                0, total_tokens + 1, dtype=torch.int32, device=DEVICE
            )
        else:
            initial_state = torch.randn(
                num_seqs,
                H_V,
                D_HEAD,
                D_HEAD,
                dtype=torch.float32,
                device=DEVICE,
            )
            ssm_state_indices = None
            cu_seqlens_cpu = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(seq_lens), 0))
            )
            cu_seqlens = cu_seqlens_cpu.to(dtype=torch.long, device=DEVICE)

        # Warmup
        for _ in range(5):
            _ = mixed_mode_dispatch(
                q,
                k,
                v,
                g,
                beta,
                A_log,
                a,
                b,
                dt_bias,
                cu_seqlens,
                initial_state,
                ssm_state_indices,
            )

        # 3. Measurement
        iters = 50
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iters):
            flusher.flush()
            _, _ = mixed_mode_dispatch(
                q,
                k,
                v,
                g,
                beta,
                A_log,
                a,
                b,
                dt_bias,
                cu_seqlens,
                initial_state,
                ssm_state_indices,
            )
        
        torch.cuda.synchronize()
        latency = (time.time() - start) / iters * 1000
        
        results.append({
            "Config": f"{n_p}P/{n_d}D",
            "Tokens": total_tokens,
            "Prefill Seqs": n_p,
            "Prefill Seq Len": p_len,
            "Decode Seqs": n_d,
            "Latency (ms)": round(latency, 2),
            "Throughput": round(total_tokens / (latency / 1000), 0)
        })

    return pd.DataFrame(results)

# Testing scenarios
test_cases = [
    {"n_prefill": 8, "avg_prefill_len": 1024, "n_decode": 0}, # Pure Prefill
    {"n_prefill": 4, "avg_prefill_len": 1024, "n_decode": 512}, # Mixed
    {"n_prefill": 0, "avg_prefill_len": 0, "n_decode": 512},  # Pure Decode
]

print("--- Qwen3.5-9B Mixed-Mode Benchmark (With Auto-Routing) ---")
print(run_mixed_benchmark(test_cases).to_string(index=False))