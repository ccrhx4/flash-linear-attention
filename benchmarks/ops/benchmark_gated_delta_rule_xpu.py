# Install and load the XPU extension that provides torch.ops._xpu_C.gdn_attention

import sys
from importlib import import_module
from pathlib import Path

import torch
from benchmark import benchmark_forward


def time_fwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean


def gdn_attention_xpu(
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    activation: str,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    has_initial_state: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices_tensor: torch.Tensor,
    num_actual_tokens: int,
    tp_size: int,
):
    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=projected_states_qkvz.dtype,
        device=projected_states_qkvz.device,
    )
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
    )

    return core_attn_out, z


repeats = 256
dtype = torch.bfloat16


def ensure_gdn_attention_registered() -> None:
    if hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "gdn_attention"):
        return

    try:
        # Importing vllm._custom_ops triggers current_platform.import_kernels().
        import_module("vllm._custom_ops")
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[4]
        vllm_repo = repo_root / "applications.ai.gpu.vllm-xpu"
        if vllm_repo.exists():
            sys.path.insert(0, str(vllm_repo))
            import_module("vllm._custom_ops")

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    raise RuntimeError("XPU device is required for benchmark_delta_rule_xpu.py")

ensure_gdn_attention_registered()

if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "gdn_attention"):
    raise RuntimeError("torch.ops._xpu_C.gdn_attention is not registered")

device = torch.device("xpu")

bs_seqlen_vals = [(8, 2048), (4, 4096), (2, 8192)]
headdim_vals = [64, 128, 256]
dim = 2048
conv_kernel_size = 4
activation = "silu"
tp_size = 1

methods = ["_xpu_C.gdn_attention"]
time_f = {}

for headdim in headdim_vals:
    for B, seqlen in bs_seqlen_vals:
        num_tokens = B * seqlen
        num_k_heads = dim // headdim
        num_v_heads = num_k_heads
        head_k_dim = headdim
        head_v_dim = headdim

        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim

        projected_states_qkvz = torch.randn(
            num_tokens,
            key_dim * 2 + value_dim * 2,
            device=device,
            dtype=dtype,
        )
        projected_states_ba = torch.randn(
            num_tokens,
            num_v_heads * 2,
            device=device,
            dtype=dtype,
        )

        conv_state = torch.zeros(
            B,
            conv_kernel_size - 1,
            conv_dim,
            device=device,
            dtype=dtype,
        )
        ssm_state = torch.zeros(
            B,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device=device,
            dtype=torch.float32,
        )
        conv_weights = torch.randn(
            conv_dim,
            conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        conv_bias = None

        A_log = torch.randn(num_v_heads, device=device, dtype=torch.float32)
        dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32)

        has_initial_state = torch.zeros(B, device=device, dtype=torch.bool)
        non_spec_query_start_loc = torch.arange(
            0,
            (B + 1) * seqlen,
            seqlen,
            device=device,
            dtype=torch.int32,
        )
        non_spec_state_indices_tensor = torch.arange(B, device=device, dtype=torch.int32)

        fwd = time_fwd(
            gdn_attention_xpu,
            projected_states_qkvz,
            projected_states_ba,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state,
            ssm_state,
            conv_weights,
            conv_bias,
            activation,
            A_log,
            dt_bias,
            B,
            0,
            has_initial_state,
            non_spec_query_start_loc,
            non_spec_state_indices_tensor,
            num_tokens,
            tp_size,
            repeats=repeats,
            verbose=False,
        )
        time_f[(headdim, B, seqlen), "_xpu_C.gdn_attention"] = fwd

        print(f"### headdim={headdim}, B={B}, seqlen={seqlen} ###")
        for method in methods:
            print(f"{method:>50} fwd:\t {time_f[(headdim, B, seqlen), method] * 1000:>6.4f} ms")
