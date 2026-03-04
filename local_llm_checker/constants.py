"""Constants for GPU bandwidth, quantization, and estimation parameters."""

_GiB = 1024**3

# GPU memory bandwidth in GB/s (theoretical peak)
# Key: substring matched against GPU name (case-insensitive)
GPU_BANDWIDTH: dict[str, float] = {
    # NVIDIA Consumer - RTX 50 series
    "RTX 5090": 1792.0,
    "RTX 5080": 960.0,
    "RTX 5070 Ti": 896.0,
    "RTX 5070": 672.0,
    # NVIDIA Consumer - RTX 40 series
    "RTX 4090": 1008.0,
    "RTX 4080 SUPER": 736.0,
    "RTX 4080": 716.8,
    "RTX 4070 Ti SUPER": 672.0,
    "RTX 4070 Ti": 504.0,
    "RTX 4070 SUPER": 504.0,
    "RTX 4070": 504.0,
    "RTX 4060 Ti": 288.0,
    "RTX 4060": 272.0,
    # NVIDIA Consumer - RTX 30 series
    "RTX 3090 Ti": 1008.0,
    "RTX 3090": 936.2,
    "RTX 3080 Ti": 912.4,
    "RTX 3080": 760.3,
    "RTX 3070 Ti": 608.3,
    "RTX 3070": 448.0,
    "RTX 3060 Ti": 448.0,
    "RTX 3060": 360.0,
    # NVIDIA Consumer - RTX 20 series
    "RTX 2080 Ti": 616.0,
    "RTX 2080 SUPER": 496.0,
    "RTX 2080": 448.0,
    "RTX 2070 SUPER": 448.0,
    "RTX 2070": 448.0,
    "RTX 2060 SUPER": 448.0,
    "RTX 2060": 336.0,
    # NVIDIA Consumer - GTX 16 series
    "GTX 1660 Ti": 288.0,
    "GTX 1660 SUPER": 336.0,
    "GTX 1660": 192.0,
    "GTX 1650": 128.0,
    # NVIDIA Data Center
    "H100": 3350.0,
    "H200": 4800.0,
    "A100 80GB": 2039.0,
    "A100 40GB": 1555.0,
    "A100": 1555.0,
    "A6000": 768.0,
    "A5000": 768.0,
    "A4000": 448.0,
    "L40S": 864.0,
    "L40": 864.0,
    "L4": 300.0,
    "T4": 320.0,
    "V100": 900.0,
    "P100": 732.0,
    # AMD
    "RX 7900 XTX": 960.0,
    "RX 7900 XT": 800.0,
    "RX 7800 XT": 624.0,
    "RX 7700 XT": 432.0,
    "RX 7600": 288.0,
    "RX 6950 XT": 576.0,
    "RX 6900 XT": 512.0,
    "RX 6800 XT": 512.0,
    "RX 6800": 512.0,
    "RX 6700 XT": 384.0,
    "MI300X": 5300.0,
    "MI250X": 3276.0,
    "MI210": 1638.0,
    # Apple Silicon (unified memory bandwidth)
    "M1 Ultra": 800.0,
    "M1 Max": 400.0,
    "M1 Pro": 200.0,
    "M1": 68.25,
    "M2 Ultra": 800.0,
    "M2 Max": 400.0,
    "M2 Pro": 200.0,
    "M2": 100.0,
    "M3 Ultra": 800.0,
    "M3 Max": 400.0,
    "M3 Pro": 150.0,
    "M3": 100.0,
    "M4 Ultra": 819.2,
    "M4 Max": 546.0,
    "M4 Pro": 273.0,
    "M4": 120.0,
}

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
}

# Quality penalty for each quantization type (fraction of quality lost)
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
    "Q3_K_M": 0.08,
    "Q3_K_S": 0.12,
    "Q3_K_L": 0.075,
    "Q2_K": 0.25,
    "IQ4_XS": 0.05,
    "IQ3_XXS": 0.18,
    "IQ2_XXS": 0.30,
}

# Preferred quantization types ordered from best to acceptable
QUANT_PREFERENCE_ORDER = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q3_K_M",
    "Q3_K_L",
    "Q8_0",
    "IQ4_XS",
    "Q4_0",
    "Q5_0",
    "Q3_K_S",
    "F16",
    "BF16",
    "Q2_K",
    "IQ3_XXS",
    "IQ2_XXS",
]

# Framework overhead in bytes (~500MB)
FRAMEWORK_OVERHEAD_BYTES = 500_000_000

# Minimum compute capability for common frameworks
MIN_COMPUTE_CAPABILITY_OLLAMA = (5, 0)
MIN_COMPUTE_CAPABILITY_VLLM = (7, 0)

# NVIDIA GPU compute capability lookup (substring match, case-insensitive)
NVIDIA_COMPUTE_CAPABILITY: dict[str, tuple[int, int]] = {
    # RTX 50 series (Blackwell)
    "RTX 5090": (10, 0),
    "RTX 5080": (10, 0),
    "RTX 5070": (10, 0),
    # RTX 40 series (Ada Lovelace)
    "RTX 4090": (8, 9),
    "RTX 4080": (8, 9),
    "RTX 4070": (8, 9),
    "RTX 4060": (8, 9),
    # RTX 30 series (Ampere)
    "RTX 3090": (8, 6),
    "RTX 3080": (8, 6),
    "RTX 3070": (8, 6),
    "RTX 3060": (8, 6),
    # RTX 20 series (Turing)
    "RTX 2080": (7, 5),
    "RTX 2070": (7, 5),
    "RTX 2060": (7, 5),
    # GTX 16 series (Turing)
    "GTX 1660": (7, 5),
    "GTX 1650": (7, 5),
    # GTX 10 series (Pascal)
    "GTX 1080": (6, 1),
    "GTX 1070": (6, 1),
    "GTX 1060": (6, 1),
    # Data Center
    "H100": (9, 0),
    "H200": (9, 0),
    "A100": (8, 0),
    "A6000": (8, 6),
    "A5000": (8, 6),
    "A4000": (8, 6),
    "L40": (8, 9),
    "L4": (8, 9),
    "T4": (7, 5),
    "V100": (7, 0),
    "P100": (6, 0),
}
