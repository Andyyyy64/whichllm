# VHS demo recording for whichllm
# Install vhs: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape
#
# This will generate assets/demo.gif which can then replace
# assets/demo.png in README.md if desired.

Output assets/demo.gif

Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "whichllm --gpu 'RTX 4090' --top 5"
Sleep 500ms
Enter
Sleep 5s

Type "whichllm --gpu 'RTX 4090' --top 5 --profile coding"
Sleep 500ms
Enter
Sleep 5s
