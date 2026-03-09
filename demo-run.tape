# VHS demo recording for whichllm run/snippet
# Install: https://github.com/charmbracelet/vhs
# Run: vhs demo-run.tape
# Output: assets/demo-run.gif

Output assets/demo-run.gif

Set FontSize 15
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

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
Type "whichllm snippet 'qwen 2.5 1.5b gguf'"
Sleep 500ms
Enter
Sleep 4s

# Scene 2: run — download and chat
Type "whichllm run 'qwen 2.5 1.5b gguf'"
Sleep 500ms
Enter
Sleep 12s

# Chat
Type "Explain quantum computing in one sentence."
Enter
Sleep 8s
Sleep 2s

Type "exit"
Enter
Sleep 2s
