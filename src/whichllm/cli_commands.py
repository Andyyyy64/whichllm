"""Stable command exports for the Typer facade."""

from whichllm.commands.recommend import main_command
from whichllm.commands.plan import plan_command
from whichllm.commands.upgrade import upgrade_command
from whichllm.commands.run import run_command
from whichllm.commands.snippet import snippet_command
from whichllm.commands.hardware import hardware_command

__all__ = [
    "main_command",
    "plan_command",
    "upgrade_command",
    "run_command",
    "snippet_command",
    "hardware_command",
]
