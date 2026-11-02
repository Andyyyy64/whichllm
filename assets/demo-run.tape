# VHS demo recording for whichllm run/snippet
# Install: https://github.com/charmbracelet/vhs
# Needs llama-cli on PATH and unsloth/Qwen3.5-2B-GGUF:Q4_K_M in its cache.
# Run: cd assets && LLAMA_ARG_THREADS=8 vhs demo-run.tape
# Output: demo-run.gif

Output demo-run.gif

Set FontSize 15
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms
Set Framerate 10

# Warm up cache
Hide
Type "whichllm --top 1 > /dev/null 2>&1"
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

# Scene 1: snippet — show the code
Type "whichllm snippet 'qwen 3.5 2b gguf'"
Sleep 500ms
Enter
Sleep 4s

# Scene 2: run — download and chat
Type "whichllm run 'qwen 3.5 2b gguf'"
Sleep 500ms
Enter
Sleep 8s

# Chat
Type "What is the capital of Japan?"
Enter
Sleep 14s

Type "/exit"
Enter
Sleep 2s
