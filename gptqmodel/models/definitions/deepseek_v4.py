from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class DeepSeekV4QModel(BaseQModel):
    require_trust_remote_code = False
    require_fast_init = False
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "wq_a:0",
                "wkv:0",
                "wq_b:1",
                "wo_a:2",
                "wo_b:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate.weight:!",),
                "experts": ("gate_up_proj:0", "down_proj:1"),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]
