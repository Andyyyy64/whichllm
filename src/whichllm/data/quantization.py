"""Quantization tiers: bytes-per-weight, quality penalty, and preference order."""

# Bytes per weight for each quantization type
QUANT_BYTES_PER_WEIGHT: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0625,
    "Q6_K": 0.8125,
    "Q5_K_M": 0.6875,
    "Q5_K_S": 0.6875,
    "Q5_0": 0.625,
    "Q4_K_M": 0.5625,
    "Q4_K_S": 0.5625,
    "Q4_0": 0.5,
    "Q3_K_M": 0.4375,
    "Q3_K_S": 0.4375,
    "Q3_K_L": 0.4375,
    "Q2_K": 0.3125,
    "IQ4_XS": 0.5,
    "IQ3_XXS": 0.375,
    "IQ2_XXS": 0.25,
    # 4-bit microscaling float formats (OCP MXFP4 / NVIDIA NVFP4).
    # MXFP4: E2M1 element + one E8M0 (8-bit) scale per block of 32 weights
    #   -> (4*32 + 8) / 32 = 4.25 bits/weight = 0.53125 bytes.
    # NVFP4: E2M1 element + one E4M3 (8-bit) scale per block of 16 weights
    #   (plus a negligible per-tensor FP32 scale) -> (4*16 + 8) / 16 = 4.5 bits
    #   = 0.5625 bytes, the same footprint as Q4_K_M.
    "MXFP4": 0.53125,
    "NVFP4": 0.5625,
    # Sub-2-bit / ternary tiers (extremely lossy)
    "Q1_0": 0.28,
    "Q2_0": 0.28,
    "TQ1_0": 0.21,
    "TQ2_0": 0.28,
    "IQ1_S": 0.21,
    "IQ1_M": 0.22,
    "IQ2_S": 0.275,
    "IQ2_M": 0.30,
    "IQ3_S": 0.40,
    "IQ3_M": 0.42,
    "IQ3_XS": 0.41,
    "IQ4_NL": 0.5,
}

# Quality penalty for each quantization type (fraction of quality lost)
# Sub-2-bit and ternary quants lose 30-60% of model quality - whichllm
# previously fell back to 5% which over-rewarded extreme quants.
QUANT_QUALITY_PENALTY: dict[str, float] = {
    "F32": 0.0,
    "F16": 0.0,
    "BF16": 0.0,
    "Q8_0": 0.01,
    "Q6_K": 0.02,
    "Q5_K_M": 0.03,
    "Q5_K_S": 0.035,
    "Q5_0": 0.035,
    "Q4_K_M": 0.05,
    "Q4_K_S": 0.055,
    "Q4_0": 0.06,
    # NVFP4's finer per-16 E4M3 scale recovers more accuracy than MXFP4's
    # coarser per-32 E8M0 (power-of-two) scale, so NVFP4 sits on par with the
    # mature 4-bit quantizers (Q4_K_M / AWQ at 0.05) while MXFP4 is slightly
    # lossier.
    "NVFP4": 0.05,
    "MXFP4": 0.06,
    "Q3_K_M": 0.08,
    "Q3_K_S": 0.12,
    "Q3_K_L": 0.075,
    "Q2_K": 0.25,
    "IQ4_XS": 0.05,
    "IQ4_NL": 0.055,
    "IQ3_XS": 0.16,
    "IQ3_S": 0.17,
    "IQ3_M": 0.16,
    "IQ3_XXS": 0.18,
    "IQ2_M": 0.30,
    "IQ2_S": 0.32,
    "IQ2_XXS": 0.40,
    "IQ1_M": 0.50,
    "IQ1_S": 0.55,
    "Q2_0": 0.45,
    "Q1_0": 0.55,
    "TQ2_0": 0.45,
    "TQ1_0": 0.55,
}

# Preferred quantization types ordered from best to acceptable.
# Sub-3-bit and 1-bit ternary variants sit at the tail so they are only
# selected when nothing else is available or when explicitly requested.
QUANT_PREFERENCE_ORDER = [
    "Q4_K_M",
    "Q4_K_S",
    "NVFP4",
    "MXFP4",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q3_K_M",
    "Q3_K_L",
    "Q8_0",
    "IQ4_XS",
    "IQ4_NL",
    "Q4_0",
    "Q5_0",
    "Q3_K_S",
    "F16",
    "BF16",
    "IQ3_M",
    "IQ3_S",
    "IQ3_XS",
    "Q2_K",
    "IQ3_XXS",
    "IQ2_M",
    "IQ2_S",
    "IQ2_XXS",
    "IQ1_M",
    "IQ1_S",
    "Q2_0",
    "TQ2_0",
    "Q1_0",
    "TQ1_0",
]
