"""Framework overhead and minimum compute capability thresholds."""

# Framework overhead in bytes (~500MB)
FRAMEWORK_OVERHEAD_BYTES = 500_000_000

# Minimum compute capability for common frameworks
MIN_COMPUTE_CAPABILITY_OLLAMA = (5, 0)
MIN_COMPUTE_CAPABILITY_VLLM = (7, 0)
