"""Compatibility shim: per-surface output modules now live alongside this file.

This module re-exports the public ``display_*`` functions so existing imports
(``from whichllm.output.display import display_ranking``) keep working. New
code should import from the specific submodule:

- ``whichllm.output.ranking`` for ranking + hardware tables
- ``whichllm.output.plan`` for the plan command
- ``whichllm.output.upgrade`` for the upgrade comparison
- ``whichllm.output.json_output`` for machine-readable JSON output
- ``whichllm.output.formatting`` for shared byte/param/date/color helpers
- ``whichllm.output._console`` for the shared Rich ``Console`` instance

The shared ``console`` symbol is re-exported here for read access. Code that
needs to *replace* the console (e.g. test capture) should set
``whichllm.output._console.console`` so every surface picks up the change.
"""

from whichllm.output._console import console
from whichllm.output.json_output import (
    display_json,
    display_plan_json,
    display_upgrade_json,
)
from whichllm.output.plan import display_plan
from whichllm.output.ranking import display_hardware, display_ranking
from whichllm.output.upgrade import display_upgrade

__all__ = [
    "console",
    "display_hardware",
    "display_json",
    "display_plan",
    "display_plan_json",
    "display_ranking",
    "display_upgrade",
    "display_upgrade_json",
]
