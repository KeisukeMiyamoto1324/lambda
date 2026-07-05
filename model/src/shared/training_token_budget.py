from rich.table import Table

from src.shared.console import console


def show_training_token_budget(
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    device_count: int,
    max_len: int,
    parameter_count: int,
) -> None:
    # ---------------------------------------------------------
    # Show the fixed context-token budget that Lightning will
    # consume when it reaches the configured optimizer step count.
    # ---------------------------------------------------------
    global_batch_size = batch_size * device_count
    global_effective_batch_size = global_batch_size * gradient_accumulation_steps
    planned_tokens = max_steps * global_effective_batch_size * max_len
    tokens_per_parameter = planned_tokens / parameter_count

    table = Table(title="Planned Training Tokens", show_header=True)
    table.add_column("Item")
    table.add_column("Value", justify="right")
    table.add_row("Max steps", f"{max_steps:,}")
    table.add_row("Context length", f"{max_len:,}")
    table.add_row("Global batch size", f"{global_batch_size:,}")
    table.add_row("Gradient accumulation", f"{gradient_accumulation_steps:,}")
    table.add_row("Global effective batch size", f"{global_effective_batch_size:,}")
    table.add_row("Model parameters", f"{parameter_count:,}")
    table.add_row("Planned tokens", f"{planned_tokens:,}")
    table.add_row("Tokens per parameter", f"{tokens_per_parameter:,.2f}x")
    console.print(table)
