"""Fill a model's MoE fields from its config.json shape.

The Hub parser reads expert counts from the top level of config.json, which
misses newer layouts that keep them under ``text_config`` (Qwen3.8-Flash-Next:
512 experts, 10 active). Such a model was treated as dense, so a 180B MoE was
estimated at the speed of reading 180B parameters per token, and expert
offload was never considered. The shape parser does read nested configs.
"""

from __future__ import annotations

from whichllm.models.types import ModelInfo

# A checkpoint is never all experts (matches engine.moe).
_MAX_EXPERT_FRACTION = 0.97


def fill_moe_from_shape(model: ModelInfo) -> bool:
    """Mark ``model`` as MoE and estimate active parameters; True if changed."""
    shape = model.shape
    if shape is None or not shape.num_experts or shape.num_experts < 2:
        return False
    if model.is_moe and model.parameter_count_active:
        return False

    from whichllm.engine.moe import active_expert_fraction, routed_expert_params

    model.is_moe = True
    if not model.parameter_count_active and model.parameter_count > 0:
        routed = routed_expert_params(model)
        if routed is not None:
            routed = min(routed, int(model.parameter_count * _MAX_EXPERT_FRACTION))
            fraction = active_expert_fraction(model)
            model.parameter_count_active = int(
                model.parameter_count - routed + routed * fraction
            )
    return True
