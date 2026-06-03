"""Canonical Rich Console instance shared by every output surface.

Tests patch the ``console`` attribute on this module to capture output
(e.g. ``whichllm.output._console.console = Console(file=buf, ...)``).
Surface modules look up the console via this module so the patch
propagates without each module holding its own binding.
"""

from rich.console import Console

console = Console()
