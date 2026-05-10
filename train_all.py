"""
train_all.py
============
Master training script — runs the full pipeline in order.
Displays hardware utilisation, overall progress, estimated time.
Usage: python train_all.py <path_to_csv> [--granularity month|week]
"""

import sys
import time
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# ── Rich progress display ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.table   import Table
    from rich.progress import (Progress, SpinnerColumn, BarColumn,
                               TextColumn, TimeElapsedColumn, TimeRemainingColumn)
    from rich.layout  import Layout
    from rich.live    import Live
    from rich.text    import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Tip: pip install rich  →  for beautiful progress bars")


def print_header():
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold red]🚔  LAPD Crime Intelligence System[/bold red]\n"
            "[dim]Pipeline: Data Cleaning → EDA → Hotspot Model → "
            "Crime Type Model → Trend Forecaster[/dim]",
            border_style="red",
        ))
    else:
        print("=" * 70)
        print("  LAPD Crime Intelligence System — Full Training Pipeline")
        print("=" * 70)


def get_hardware_info() -> dict:
    info = {"platform": platform.processor(), "python": platform.python_version()}
    try:
        import psutil
        info["ram_gb"]      = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_used_pct"]= psutil.virtual_memory().percent
        info["cpu_pct"]     = psutil.cpu_percent(interval=0.5)
    except ImportError:
        pass

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            info["gpu_name"]     = gpus[0].name
            info["gpu_mem_mb"]   = gpus[0].memoryTotal
            info["gpu_used_pct"] = gpus[0].load * 100
    except Exception:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                info["gpu_name"]     = parts[0].strip()
                info["gpu_mem_mb"]   = int(parts[1].strip())
                info["gpu_used_pct"] = float(parts[2].strip())
        except Exception:
            pass

    return info


def print_hardware_table(info: dict):
    if not RICH_AVAILABLE:
        print(f"\nHardware: {info}\n")
        return

    table = Table(title="🖥️  Hardware Configuration", border_style="dim")
    table.add_column("Component", style="cyan")
    table.add_column("Value",     style="white")

    table.add_row("CPU",  info.get("platform", "Unknown"))
    table.add_row("RAM",  f"{info.get('ram_gb','?')} GB "
                          f"({info.get('ram_used_pct','?')}% used)")
    if "gpu_name" in info:
        table.add_row("GPU",  info["gpu_name"])
        table.add_row("VRAM", f"{info.get('gpu_mem_mb','?')} MB")
    table.add_row("Python", info.get("python", "Unknown"))
    console.print(table)


def run_stage(name: str, fn, *args, **kwargs):
    """Run a pipeline stage with timing."""
    if RICH_AVAILABLE:
        console.rule(f"[bold red]Stage: {name}[/bold red]")
    else:
        print(f"\n{'='*60}")
        print(f"  STAGE: {name}")
        print(f"{'='*60}")

    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start

    if RICH_AVAILABLE:
        console.print(f"[green]✅  {name} completed in {elapsed:.1f}s[/green]\n")
    else:
        print(f"\n✅  {name} completed in {elapsed:.1f}s\n")

    return result, elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="LAPD Crime ML Pipeline")
    parser.add_argument("csv_path",      type=str, help="Path to raw CSV file")
    parser.add_argument("--granularity", type=str, default="month",
                        choices=["month","week"],
                        help="Time granularity for hotspot model (default: month)")
    parser.add_argument("--skip-trends", action="store_true",
                        help="Skip Prophet trend fitting (faster)")
    parser.add_argument("--skip-type",   action="store_true",
                        help="Skip crime type model")
    args = parser.parse_args()

    print_header()

    # Hardware check
    hw = get_hardware_info()
    print_hardware_table(hw)

    pipeline_start = time.time()
    stage_times    = {}

    # ── STAGE 1: Data Pipeline ────────────────────────────────────────────────
    from data_pipeline import run_pipeline
    df, t = run_stage("Data Cleaning & EDA", run_pipeline, args.csv_path)
    stage_times["Data Pipeline"] = t

    # ── STAGE 2: Hotspot Model ────────────────────────────────────────────────
    from model_hotspot import run_hotspot_pipeline
    _, t = run_stage(f"Hotspot Model ({args.granularity})",
                     run_hotspot_pipeline, args.granularity)
    stage_times["Hotspot Model"] = t

    # ── STAGE 3: Crime Type Model ─────────────────────────────────────────────
    if not args.skip_type:
        from model_crime_type import run_crime_type_pipeline
        _, t = run_stage("Crime Type Model", run_crime_type_pipeline)
        stage_times["Crime Type Model"] = t

    # ── STAGE 4: Trends ───────────────────────────────────────────────────────
    if not args.skip_trends:
        from model_trends import run_trends_pipeline
        _, t = run_stage("Trend Forecaster", run_trends_pipeline)
        stage_times["Trend Forecaster"] = t

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start

    if RICH_AVAILABLE:
        summary = Table(title="🎉  Pipeline Summary", border_style="green")
        summary.add_column("Stage",   style="cyan")
        summary.add_column("Time",    style="white")
        summary.add_column("Status",  style="green")
        for stage, secs in stage_times.items():
            summary.add_row(stage, f"{secs:.1f}s", "✅ Complete")
        summary.add_row("[bold]TOTAL[/bold]", f"[bold]{total:.1f}s[/bold]", "")
        console.print(summary)

        console.print(Panel(
            "[bold green]All models trained and saved![/bold green]\n\n"
            "Next step → launch the dashboard:\n"
            "[yellow]  streamlit run app.py[/yellow]",
            border_style="green",
        ))
    else:
        print("\n" + "="*50)
        print(f"  Pipeline complete in {total:.1f}s")
        for stage, secs in stage_times.items():
            print(f"  {stage:<25} {secs:.1f}s")
        print("\nRun:  streamlit run app.py")
        print("="*50)


if __name__ == "__main__":
    main()