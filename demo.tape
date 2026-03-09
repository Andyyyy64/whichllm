# VHS demo recording for whichllm
# Install: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape
# Output: assets/demo.gif

Output assets/demo.gif

Set FontSize 15
Set Width 1200
Set Height 1100
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

# Warm up cache invisibly
Hide
Type "whichllm --top 1 > /dev/null 2>&1"
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

# Scene 1: Just run it
Type "whichllm"
Sleep 500ms
Enter
Sleep 6s
Sleep 2s

# Scene 2: What GPU do I need?
Type "whichllm plan 'qwen 72b'"
Sleep 500ms
Enter
Sleep 6s
Sleep 3s
