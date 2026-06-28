from rich.panel import Panel
from rich.table import Table

from src.shared.console import console


def estimate_training_tokens(
    max_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    device_count: int,
    max_len: int,
) -> int:
    # ---------------------------------------------------------
    # Estimate token budget from optimizer updates and global
    # effective batch size used by the Lightning trainer.
    # ---------------------------------------------------------
    global_effective_batch_size = batch_size * gradient_accumulation_steps * device_count
    return max_steps * global_effective_batch_size * max_len


def format_token_billions(token_count: int) -> str:
    # ---------------------------------------------------------
    # Format large training budgets in billions because model
    # scaling plans are usually compared in B-token units.
    # ---------------------------------------------------------
    return f"{token_count / 1_000_000_000:.3f}B"


def show_training_token_plan(
    stage_name: str,
    max_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    device_count: int,
    max_len: int,
) -> int:
    # ---------------------------------------------------------
    # Print the estimated training token budget before training
    # starts, while returning the raw value for saved metadata.
    # ---------------------------------------------------------
    global_effective_batch_size = batch_size * gradient_accumulation_steps * device_count
    estimated_tokens = estimate_training_tokens(
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        device_count=device_count,
        max_len=max_len,
    )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("Stage", stage_name)
    table.add_row("Max steps", f"{max_steps:,}")
    table.add_row("Global effective batch", f"{global_effective_batch_size:,}")
    table.add_row("Max length", f"{max_len:,}")
    table.add_row("Estimated training tokens", f"{format_token_billions(estimated_tokens)}")

    console.print(Panel(table, title="Training Token Plan", border_style="cyan"))
    return estimated_tokens
