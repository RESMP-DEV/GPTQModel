# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MiMoV2QModel(BaseQModel):
    # MiMo-V2.5-Pro: 1.02T MoE (384 routed experts, 8 per token)
    # Layer 0: dense MLP; layers 1-69: MoE (per moe_layer_freq config)
    # Hybrid attention: 10 global + 60 sliding-window layers (per hybrid_layer_pattern)
    # Fused QKV projection (attention_projection_layout == "fused_qkv")
    # 3 MTP layers stored separately in model_mtp.safetensors

    require_trust_remote_code = True

    require_fast_init = False

    layer_modules_strict = False

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    out_of_model_tensors = {"files": ["model_mtp.safetensors"]}

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("qkv_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # dense MLP fallback for layer 0 (moe_layer_freq[0] == 0)
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]
