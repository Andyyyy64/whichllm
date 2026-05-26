"""GPU bandwidth, NVIDIA compute capability, and AMD shared-memory APU markers."""

_GiB = 1024**3

AMD_SHARED_MEMORY_APU_MARKERS: tuple[str, ...] = (
    "STRIX HALO",
    "STRXLGEN",
    "RADEON 8050S",
    "RADEON 8060S",
    "RADEON 890M",
    "RADEON 880M",
    "RADEON 860M",
    "RADEON 840M",
    "RADEON 780M",
    "RADEON 760M",
    "RADEON 740M",
    "RADEON 680M",
    "RADEON 660M",
    "RYZEN AI 9",
    "RYZEN AI 7",
    "RYZEN AI 5",
    "RYZEN AI MAX",
)

# GPU memory bandwidth in GB/s (theoretical peak)
# Key: substring matched against GPU name (case-insensitive)
GPU_BANDWIDTH: dict[str, float] = {
    # NVIDIA Consumer - RTX 50 series
    "RTX 5090": 1792.0,
    "RTX 5080": 960.0,
    "RTX 5070 Ti": 896.0,
    "RTX 5070": 672.0,
    "RTX 5060 Ti": 448.0,
    "RTX 5060": 336.0,
    "RTX 3050": 224.0,
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
    "DGX Spark": 273.0,
    "GB10": 273.0,
    "A100 80GB": 2039.0,
    "A100 40GB": 1555.0,
    "A100": 1555.0,
    "RTX A3000 Laptop": 264.0,
    "A6000": 768.0,
    "A5000": 768.0,
    "A4000": 448.0,
    "L40S": 864.0,
    "L40": 864.0,
    "L4": 300.0,
    "T4": 320.0,
    "V100": 900.0,
    "P100": 732.0,
    # NVIDIA Kepler (legacy, Vulkan-only — no CUDA in modern llama.cpp).
    # Values are theoretical peak memory bandwidth (GB/s) from NVIDIA
    # datasheets. Kepler (compute capability 3.x) was dropped by CUDA 12 and
    # current llama.cpp CUDA builds, so these cards run via the Vulkan backend
    # on Linux. See VULKAN_ONLY_GPUS below.
    "Quadro K6000": 288.0,
    "Quadro K5200": 192.3,
    "Quadro K4200": 173.0,
    "Quadro K2200": 80.0,
    "Quadro K620": 29.0,
    "Quadro K420": 14.4,
    "GTX 780": 288.4,
    "GTX 770": 224.3,
    "GTX 760": 192.2,
    # AMD
    "RX 9070 XT": 640.0,
    "RX 9070": 560.0,
    "RX 9060 XT": 320.0,
    "RX 7900 XTX": 960.0,
    "RX 7900 XT": 800.0,
    "RX 7800 XT": 624.0,
    "RX 7700 XT": 432.0,
    "RX 7600": 288.0,
    "RX 6950 XT": 576.0,
    "RX 6900 XT": 512.0,
    "RX 6800 XT": 512.0,
    "RX 6800": 512.0,
    "RX 6750 XT": 432.0,
    "RX 6700 XT": 384.0,
    "RX 6700": 320.0,
    "RX 6650 XT": 256.0,
    "RX 6600 XT": 256.0,
    "RX 6600": 224.0,
    # AMD APUs / shared-memory graphics
    "Ryzen AI MAX+ 395": 256.0,
    "Ryzen AI MAX 395": 256.0,
    "Radeon 890M": 120.0,
    "Radeon 880M": 120.0,
    "Radeon 860M": 90.0,
    "Radeon 840M": 60.0,
    "Radeon 780M": 90.0,
    "Radeon 760M": 75.0,
    "Radeon 740M": 60.0,
    "Radeon 680M": 75.0,
    "Radeon 660M": 55.0,
    "Radeon 8060S": 256.0,
    "Radeon 8050S": 256.0,
    "Strix Halo": 256.0,
    "STRXLGEN": 256.0,
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
    "M5 Max": 614.0,
    "M5 Pro": 307.0,
    "M5": 153.0,
}

# NVIDIA GPU compute capability lookup (substring match, case-insensitive)
NVIDIA_COMPUTE_CAPABILITY: dict[str, tuple[int, int]] = {
    # RTX 50 series (Blackwell)
    "RTX 5090": (10, 0),
    "RTX 5080": (10, 0),
    "RTX 5070": (10, 0),
    "RTX 5070 Ti": (10, 0),
    "RTX 5060": (10, 0),
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
    "DGX Spark": (12, 1),
    "GB10": (12, 1),
    "A100": (8, 0),
    "RTX A3000 Laptop": (8, 6),
    "A6000": (8, 6),
    "A5000": (8, 6),
    "A4000": (8, 6),
    "L40": (8, 9),
    "L4": (8, 9),
    "T4": (7, 5),
    "V100": (7, 0),
    "P100": (6, 0),
    # Kepler series (compute capability 3.x) — legacy, Vulkan-only.
    # CUDA 12 and current llama.cpp CUDA builds dropped Kepler support, so
    # these cards only run through the Vulkan backend. See VULKAN_ONLY_GPUS.
    "Quadro K6000": (3, 5),
    "Quadro K5200": (3, 5),
    "Quadro K4200": (3, 0),
    "Quadro K2200": (3, 0),
    "Quadro K620": (3, 0),
    "Quadro K420": (3, 0),
    "GTX 780": (3, 5),
    "GTX 770": (3, 0),
    "GTX 760": (3, 0),
}

# Legacy NVIDIA GPUs with no CUDA support in modern llama.cpp builds.
# Kepler (compute capability 3.0/3.5) was removed in CUDA 12, so these cards
# can only be used through the Vulkan backend (Linux) in current llama.cpp.
# Entries are substrings matched case-insensitively against the GPU name, the
# same convention used by GPU_BANDWIDTH and NVIDIA_COMPUTE_CAPABILITY.
VULKAN_ONLY_GPUS: frozenset[str] = frozenset(
    {
        "Quadro K6000",
        "Quadro K5200",
        "Quadro K4200",
        "Quadro K2200",
        "Quadro K620",
        "Quadro K420",
        "GTX 780",
        "GTX 770",
        "GTX 760",
    }
)
