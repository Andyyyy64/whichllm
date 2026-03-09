#!/usr/bin/env bash
# Alternative demo recording script using asciinema + agg (or similar).
# Use this if you don't have vhs installed.
#
# Prerequisites (pick one):
#   Option A: vhs        -> just run: vhs demo.tape
#   Option B: asciinema  -> run this script, then convert with agg
#
# Converting asciinema recording to GIF:
#   pip install asciinema
#   cargo install agg    # or: brew install agg
#   asciinema rec assets/demo.cast  (this script does that)
#   agg assets/demo.cast assets/demo.gif

set -euo pipefail

CAST_FILE="assets/demo.cast"
GIF_FILE="assets/demo.gif"

if command -v vhs &>/dev/null; then
    echo "vhs is installed. Running: vhs demo.tape"
    vhs demo.tape
    exit 0
fi

if ! command -v asciinema &>/dev/null; then
    echo "Neither vhs nor asciinema is installed."
    echo ""
    echo "Install one of:"
    echo "  vhs:       https://github.com/charmbracelet/vhs"
    echo "  asciinema: pip install asciinema"
    exit 1
fi

echo "Recording demo with asciinema..."
echo "Run the following commands when the recording starts:"
echo ""
echo "  whichllm --gpu 'RTX 4090' --top 5"
echo "  whichllm --gpu 'RTX 4090' --top 5 --profile coding"
echo "  exit"
echo ""

asciinema rec "$CAST_FILE"

echo ""
echo "Recording saved to $CAST_FILE"
echo "Convert to GIF with: agg $CAST_FILE $GIF_FILE"
