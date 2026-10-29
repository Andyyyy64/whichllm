"""Routed-expert share of mixture-of-experts checkpoints.

With the layer shape from config.json the routed experts are counted exactly:
three projections (gate, up, down) of ``hidden x expert_ffn`` per expert per
MoE layer. Shared experts, attention and embeddings stay with the dense part,
which is what runtimes keep on the GPU when experts move to system RAM.
"""

from __future__ import annotations

from whichllm.models.shape import ModelShape
from whichllm.models.types import ModelInfo

# A checkpoint is never all experts; the cap also absorbs parameter counts
# that undercount packed 4-bit weights.
_MAX_EXPERT_FRACTION = 0.97


def moe_layer_count(shape: ModelShape) -> int:
    """Layers that carry routed experts (DeepSeek-style models start dense)."""
    return max(shape.num_layers - shape.first_dense_layers, 0)


def routed_expert_params(model: ModelInfo) -> int | None:
    """Routed-expert parameters from the model shape, or None if unknown."""
    shape = model.shape
    if shape is None or not shape.num_experts or not shape.hidden_size:
        return None
    expert_ffn = shape.moe_intermediate_size or shape.intermediate_size
    layers = moe_layer_count(shape)
    if not expert_ffn or not layers:
        return None
    return layers * shape.num_experts * 3 * shape.hidden_size * expert_ffn


def routed_expert_fraction(model: ModelInfo) -> float:
    """Share of the stored weights that are routed experts (0.0 for dense)."""
    if not model.is_moe or model.parameter_count <= 0:
        return 0.0
    routed = routed_expert_params(model)
    if routed is not None:
        return min(routed / model.parameter_count, _MAX_EXPERT_FRACTION)
    if model.parameter_count_active:
        # Without the shape, count only the weights no token reads as
        # experts. That understates the movable share, so placement stays
        # on the cautious side.
        inactive = 1.0 - model.parameter_count_active / model.parameter_count
        return max(0.0, min(inactive, _MAX_EXPERT_FRACTION))
    return 0.0


def active_expert_fraction(model: ModelInfo) -> float:
    """Share of the routed-expert weights read for each generated token."""
    shape = model.shape
    if shape and shape.num_experts and shape.experts_per_token:
        return min(shape.experts_per_token / shape.num_experts, 1.0)
    if model.parameter_count_active and model.parameter_count > 0:
        return min(model.parameter_count_active / model.parameter_count, 1.0)
    return 1.0
