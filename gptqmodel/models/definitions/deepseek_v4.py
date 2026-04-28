from typing import Optional

import torch

from ..base import BaseQModel
from ..moe_lifecycle import W1W3W2MoELifecycleHooks


class DeepSeekV4QModel(BaseQModel):
    require_trust_remote_code = False
    require_fast_init = False
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"
    moe_lifecycle_hooks = W1W3W2MoELifecycleHooks()
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
                "gate": ("gate:!",),
                "experts": {
                    "#": ("w1:0", "w3:0", "w2:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def shell_module_materialize(
        self,
        target_submodule: torch.nn.Module,
        device: torch.device,
        non_blocking: bool = False,
        role: str = "default",
        named_module: Optional["NamedModule"] = None,
    ) -> torch.nn.Module:
        try:
            from v4_register import _PackedLinear
        except ImportError:
            _PackedLinear = None

        if _PackedLinear is not None and isinstance(target_submodule, _PackedLinear):
            if target_submodule.weight.device.type == "meta":
                self.turtle_model.materialize_submodule(
                    target_model=self.model,
                    target_submodule=target_submodule,
                    device=torch.device("cpu"),
                )
            result = target_submodule.dequantize_to_linear()
            if device.type != "cpu":
                result = result.to(device)
            if named_module is not None:
                named_module.module = result
            return result

        return super().shell_module_materialize(
            target_submodule=target_submodule,
            device=device,
            non_blocking=non_blocking,
            role=role,
            named_module=named_module,
        )
