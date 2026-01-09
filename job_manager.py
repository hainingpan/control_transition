import json
import subprocess
import time
import os
import re
from collections import defaultdict

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

# --- Configuration ---
PARAMS_FILE = "params_APT_OP_T.txt"
SUBMISSION_SCRIPT = "slurm_APT_OP_T.sh"
STATE_FILE = "job_manager_state.json"
MAX_JOBS_PER_PARTITION = {
    "main": 500,
    "gpu": 150,
    "default": 150 # Used if partition is not in this dict
}
MAX_RESUBMITS = 3
CHECK_INTERVAL = 60  # secondspeasons to trigger a resubmit
RESUBMIT_REASONS = ["preemption"] # Case-insensitive list of reasons to trigger a resubmit

# --- State Management ---

def initialize_state(params_file):
    """Creates the initial state from the parameter file."""
    state = {}
    with open(params_file, 'r') as f:
        for i, line in enumerate(f):
            state[str(i + 1)] = {
                "params": line.strip(),
                "status": "NOT_SUBMITTED",
                "job_id": None,
                "submit_count": 0,
                "task_id": i + 1,
                "failure_reason": None,
                "elapsed_time": None,  # For RUNNING jobs
                "wall_time": None,     # For COMPLETED jobs
            }
    return state

def load_state(state_file, params_file):
    """Loads the state from the state file, or initializes it if it doesn't exist."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            # Ensure new fields exist for existing state files
            for job_info in state.values():
                if "elapsed_time" not in job_info:
                    job_info["elapsed_time"] = None
                if "wall_time" not in job_info:
                    job_info["wall_time"] = None
            return state
    else:
        return initialize_state(params_file)

def save_state(state, state_file):
    """Saves the state to the state file."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

# --- Slurm Interaction ---

def get_partition_from_script(script_path):
    """Parses the submission script to find the specified partition."""
    try:
        with open(script_path, 'r') as f:
            for line in f:
                if line.strip().startswith("#SBATCH --partition="):
                    return line.strip().split('=')[1]
    except FileNotFoundError:
        print(f"Error: Submission script not found at {script_path}")
    return None

def get_slurm_queue_per_partition():
    """Gets the current job counts per partition for the user."""
    partition_counts = defaultdict(int)
    try:
        user = os.environ.get("USER")
        cmd = f"squeue -u {user} -h -o \"%P\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line:
                partition_counts[line.strip()] += 1
        return partition_counts
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting Slurm queue: {e}")
        return partition_counts

def get_job_final_status(job_id):
    """Gets the final status, reason, and elapsed time of a completed job using sacct."""
    try:
        cmd = f"sacct -j {job_id} --format=State,Reason,Elapsed -n -P"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if not lines:
            return "UNKNOWN", "", None
        # Parse pipe-separated output
        parts = lines[0].split('|')
        final_state = parts[0].strip() if len(parts) > 0 else "UNKNOWN"
        reason = parts[1].strip() if len(parts) > 1 else ""
        elapsed = parts[2].strip() if len(parts) > 2 else None
        return final_state, reason, elapsed
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting job final status for {job_id}: {e}")
        return "UNKNOWN", "", None

def get_slurm_job_details():
    """Gets job IDs, states, and elapsed times for all user jobs in the queue."""
    job_details = {}
    try:
        user = os.environ.get("USER")
        cmd = f"squeue -u {user} -h -o \"%A|%T|%M\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    job_id = parts[0].strip()
                    state = parts[1].strip()
                    elapsed = parts[2].strip()
                    job_details[job_id] = {"state": state, "elapsed": elapsed}
        return job_details
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting Slurm job details: {e}")
        return job_details

def submit_job(task_id, submission_script, params_file):
    """Submits a job to Slurm and returns the job ID."""
    try:
        cmd = f"sbatch --export=ALL,REAL_TASK_ID={task_id},PARAMS_FILE={params_file} --array=1 {submission_script}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return match.group(1)
        else:
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

# --- Main Logic ---

def all_jobs_are_finished(state):
    """Checks if all jobs have reached a terminal state."""
    return all(
        job_info["status"] in ["COMPLETED", "FAILED_PERMANENTLY"]
        for job_info in state.values()
    )

def handle_failed_jobs(state):
    """Handles failed jobs, marking them for resubmission if the reason is in RESUBMIT_REASONS."""
    for job_info in state.values():
        if job_info["status"] == "FAILED":
            if job_info["submit_count"] < MAX_RESUBMITS:
                reason = job_info.get("failure_reason", "").lower()
                should_resubmit = any(resubmit_reason.lower() in reason for resubmit_reason in RESUBMIT_REASONS)

                if should_resubmit:
                    job_info["status"] = "PENDING_RESUBMISSION"
                else:
                    job_info["status"] = "FAILED_PERMANENTLY"
            else:
                job_info["status"] = "FAILED_PERMANENTLY"

def check_and_update_completed_jobs(state, slurm_job_details):
    """Check job statuses and update PENDING/RUNNING/COMPLETED states."""
    for job_info in state.values():
        if job_info["status"] in ["PENDING", "RUNNING"]:
            job_id_str = str(job_info["job_id"])
            if job_id_str in slurm_job_details:
                # Job is still in the queue - update status and elapsed time
                details = slurm_job_details[job_id_str]
                slurm_state = details["state"]
                if slurm_state == "RUNNING":
                    job_info["status"] = "RUNNING"
                    job_info["elapsed_time"] = details["elapsed"]
                elif slurm_state == "PENDING":
                    job_info["status"] = "PENDING"
                    job_info["elapsed_time"] = None
                # Handle other SLURM states (CONFIGURING, etc.) as PENDING
                elif slurm_state in ["CONFIGURING", "COMPLETING"]:
                    job_info["status"] = "RUNNING"
                    job_info["elapsed_time"] = details["elapsed"]
            else:
                # Job is no longer in the queue - check final status
                final_state, reason, elapsed = get_job_final_status(job_info["job_id"])
                if "COMPLETED" in final_state:
                    job_info["status"] = "COMPLETED"
                    job_info["wall_time"] = elapsed
                    job_info["elapsed_time"] = None
                else:
                    job_info["status"] = "FAILED"
                    job_info["failure_reason"] = reason
                    job_info["wall_time"] = elapsed

def submit_and_resubmit_jobs(state, queue_counts, submission_script, params_file):
    """Submits new jobs and resubmits failed ones based on per-partition queue capacity."""
    target_partition = get_partition_from_script(submission_script)
    if not target_partition:
        return

    current_jobs_in_partition = queue_counts.get(target_partition, 0)
    max_jobs_for_partition = MAX_JOBS_PER_PARTITION.get(target_partition, MAX_JOBS_PER_PARTITION['default'])
    available_slots = max_jobs_for_partition - current_jobs_in_partition

    if available_slots <= 0:
        return

    # Prioritize resubmissions
    for job_info in sorted(state.values(), key=lambda x: x['task_id']):
        if available_slots <= 0: break
        if job_info["status"] == "PENDING_RESUBMISSION":
            new_job_id = submit_job(job_info["task_id"], submission_script, PARAMS_FILE)
            if new_job_id:
                job_info["status"] = "PENDING"
                job_info["job_id"] = new_job_id
                job_info["submit_count"] += 1
                available_slots -= 1

    # Submit new jobs
    for job_info in sorted(state.values(), key=lambda x: x['task_id']):
        if available_slots <= 0: break
        if job_info["status"] == "NOT_SUBMITTED":
            new_job_id = submit_job(job_info["task_id"], submission_script, PARAMS_FILE)
            if new_job_id:
                job_info["status"] = "PENDING"
                job_info["job_id"] = new_job_id
                job_info["submit_count"] += 1
                available_slots -= 1

# --- Visualization ---

console = Console()

def get_status_counts(state):
    """Count jobs by status."""
    counts = defaultdict(int)
    for job_info in state.values():
        counts[job_info["status"]] += 1
    return counts

def build_progress_grid(state, width=80):
    """Build a defrag-style colored grid showing job statuses."""
    text = Text()
    sorted_tasks = sorted(state.keys(), key=int)

    for i, task_id in enumerate(sorted_tasks):
        status = state[task_id]["status"]
        if status == "COMPLETED":
            text.append("█", style="green")
        elif status == "RUNNING":
            text.append("█", style="cyan bold")
        elif status == "PENDING":
            text.append("█", style="yellow")
        elif status in ["FAILED", "FAILED_PERMANENTLY"]:
            text.append("█", style="red")
        elif status == "PENDING_RESUBMISSION":
            text.append("▒", style="yellow")
        else:  # NOT_SUBMITTED
            text.append("░", style="dim")

        # Add newline at row boundary
        if (i + 1) % width == 0:
            text.append("\n")

    return text

def build_legend():
    """Build the legend text."""
    legend = Text()
    legend.append("█", style="green")
    legend.append(" Completed  ")
    legend.append("█", style="cyan bold")
    legend.append(" Running  ")
    legend.append("█", style="yellow")
    legend.append(" Pending  ")
    legend.append("█", style="red")
    legend.append(" Failed  ")
    legend.append("░", style="dim")
    legend.append(" Not Submitted")
    return legend

def build_running_table(state):
    """Build a table of currently running jobs."""
    running_jobs = [
        job for job in state.values()
        if job["status"] == "RUNNING"
    ]

    if not running_jobs:
        return None

    table = Table(title="Running Jobs", show_header=True, header_style="bold cyan")
    table.add_column("Task", style="cyan", justify="right")
    table.add_column("Job ID", justify="right")
    table.add_column("Elapsed", style="green", justify="right")

    for job in sorted(running_jobs, key=lambda x: x["task_id"])[:20]:  # Limit to 20
        elapsed = job.get("elapsed_time") or "-"
        table.add_row(
            f"#{job['task_id']}",
            str(job["job_id"]),
            elapsed
        )

    if len(running_jobs) > 20:
        table.add_row("...", f"+{len(running_jobs) - 20} more", "")

    return table

def build_pending_table(state):
    """Build a table of pending jobs."""
    pending_jobs = [
        job for job in state.values()
        if job["status"] == "PENDING"
    ]

    if not pending_jobs:
        return None

    table = Table(title="Pending Jobs", show_header=True, header_style="bold yellow")
    table.add_column("Task", style="yellow", justify="right")
    table.add_column("Job ID", justify="right")

    # Show first 10 pending jobs
    for job in sorted(pending_jobs, key=lambda x: x["task_id"])[:10]:
        table.add_row(f"#{job['task_id']}", str(job["job_id"]))

    if len(pending_jobs) > 10:
        table.add_row("...", f"+{len(pending_jobs) - 10} more")

    return table

def build_recent_completed_table(state, limit=5):
    """Build a table of recently completed jobs (by task_id, highest first)."""
    completed_jobs = [
        job for job in state.values()
        if job["status"] == "COMPLETED" and job.get("wall_time")
    ]

    if not completed_jobs:
        return None

    # Sort by task_id descending to get "recent" completions
    completed_jobs.sort(key=lambda x: x["task_id"], reverse=True)

    table = Table(title="Recently Completed", show_header=True, header_style="bold green")
    table.add_column("Task", style="green", justify="right")
    table.add_column("Job ID", justify="right")
    table.add_column("Wall Time", style="cyan", justify="right")

    for job in completed_jobs[:limit]:
        table.add_row(
            f"#{job['task_id']}",
            str(job["job_id"]),
            job.get("wall_time") or "-"
        )

    return table

def generate_display(state):
    """Generate the full Rich display."""
    import shutil
    terminal_width = shutil.get_terminal_size().columns
    # Use most of the terminal width for the grid (subtract panel borders)
    grid_width = terminal_width - 6  # Account for panel borders

    counts = get_status_counts(state)
    total = len(state)
    completed = counts.get("COMPLETED", 0)
    running = counts.get("RUNNING", 0)
    pending = counts.get("PENDING", 0)
    failed = counts.get("FAILED", 0) + counts.get("FAILED_PERMANENTLY", 0)
    not_submitted = counts.get("NOT_SUBMITTED", 0)
    pending_resub = counts.get("PENDING_RESUBMISSION", 0)

    # Calculate progress percentage
    progress_pct = (completed / total * 100) if total > 0 else 0

    # Build header
    header = Text(f"JOB MANAGER - {time.ctime()}", style="bold white", justify="center")

    # Build progress bar text
    bar_width = 50
    filled = int(bar_width * progress_pct / 100)
    progress_bar = Text()
    progress_bar.append(f"Progress: [")
    progress_bar.append("█" * filled, style="green")
    progress_bar.append("░" * (bar_width - filled), style="dim")
    progress_bar.append(f"] {progress_pct:.1f}%")

    # Build summary
    summary = Text()
    summary.append(f"{completed}", style="green bold")
    summary.append(" Completed | ")
    summary.append(f"{running}", style="cyan bold")
    summary.append(" Running | ")
    summary.append(f"{pending}", style="yellow bold")
    summary.append(" Pending | ")
    summary.append(f"{failed}", style="red bold")
    summary.append(" Failed | ")
    summary.append(f"{not_submitted}", style="dim")
    summary.append(" Not Submitted")
    if pending_resub > 0:
        summary.append(" | ")
        summary.append(f"{pending_resub}", style="yellow")
        summary.append(" Resubmit")

    # Build the grid
    grid = build_progress_grid(state, grid_width)

    # Combine all elements
    elements = [
        Panel(header, style="bold blue"),
        "",
        progress_bar,
        "",
        Panel(grid, title="Job Status Grid", border_style="dim"),
        "",
        build_legend(),
        "",
        summary,
        "",
    ]

    # Build tables and arrange them side by side
    running_table = build_running_table(state)
    pending_table = build_pending_table(state)
    completed_table = build_recent_completed_table(state)

    # Collect tables that exist
    tables = []
    if running_table:
        tables.append(running_table)
    if pending_table:
        tables.append(pending_table)
    if completed_table:
        tables.append(completed_table)

    # Display tables side by side using Columns
    if tables:
        elements.append(Columns(tables, equal=False, expand=False))

    elements.append("")
    elements.append(Text(f"Refreshing every {CHECK_INTERVAL} seconds... (Ctrl+C to stop)", style="dim italic"))

    return Group(*elements)

# --- Main Loop ---

def run_cycle(state):
    """Run one cycle of job management. Returns True if all jobs finished."""
    try:
        # Get detailed job info from squeue
        slurm_job_details = get_slurm_job_details()

        # Update job statuses
        check_and_update_completed_jobs(state, slurm_job_details)

        # Handle failed jobs
        handle_failed_jobs(state)

        # Submit new jobs
        queue_counts = get_slurm_queue_per_partition()
        submit_and_resubmit_jobs(state, queue_counts, SUBMISSION_SCRIPT, PARAMS_FILE)

        # Save state
        save_state(state, STATE_FILE)

        return all_jobs_are_finished(state)

    except Exception as e:
        console.print(f"[red]Error in cycle: {e}[/red]")
        return False

def main():
    """The main function of the job manager."""
    if not os.path.exists(PARAMS_FILE):
        console.print(f"[red]Error: Parameter file '{PARAMS_FILE}' not found.[/red]")
        return

    state = load_state(STATE_FILE, PARAMS_FILE)

    # Initial cycle run
    console.print("[bold blue]Starting Job Manager...[/bold blue]")

    try:
        with Live(generate_display(state), console=console, refresh_per_second=0.5, screen=True) as live:
            while True:
                # Run management cycle
                finished = run_cycle(state)

                # Update display
                live.update(generate_display(state))

                if finished:
                    # Show final state for a moment before exiting
                    live.update(generate_display(state))
                    time.sleep(2)
                    break

                # Wait before next cycle
                time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        console.print("\n[yellow]Job Manager stopped by user.[/yellow]")
        save_state(state, STATE_FILE)

    console.print("[bold green]All jobs have reached a terminal state. Exiting.[/bold green]")

if __name__ == "__main__":
    main()
