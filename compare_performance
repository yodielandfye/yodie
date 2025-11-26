"""
Extract and compare performance metrics from LAMMPS logs.

Measures:
- Timesteps per second
- Million atom-steps per second (Matom-step/s)
- Wall time per step
- GPU vs CPU comparison
"""

import re
from pathlib import Path
from typing import Dict, Optional


def extract_performance_metrics(log_path: Path) -> Optional[Dict]:
    """Extract performance metrics from LAMMPS log file."""
    try:
        content = log_path.read_text()
    except:
        return None
    
    metrics = {}
    
    # Extract loop time line
    loop_match = re.search(
        r"Loop time of ([\d.]+) on (\d+) procs for ([\d]+) steps with ([\d]+) atoms",
        content
    )
    
    if loop_match:
        loop_time = float(loop_match.group(1))
        n_procs = int(loop_match.group(2))
        n_steps = int(loop_match.group(3))
        n_atoms = int(loop_match.group(4))
        
        metrics["loop_time"] = loop_time
        metrics["n_procs"] = n_procs
        metrics["n_steps"] = n_steps
        metrics["n_atoms"] = n_atoms
        metrics["timesteps_per_sec"] = n_steps / loop_time
        metrics["matom_steps_per_sec"] = (n_steps * n_atoms) / loop_time / 1_000_000
    
    return metrics if metrics else None


def compare_logs(gpu_log: Path, cpu_log: Path) -> None:
    """Compare GPU and CPU performance metrics."""
    
    print("=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    gpu_metrics = extract_performance_metrics(gpu_log)
    cpu_metrics = extract_performance_metrics(cpu_log)
    
    if not gpu_metrics:
        print(f"❌ Could not extract metrics from {gpu_log}")
        return
    
    if not cpu_metrics:
        print(f"❌ Could not extract metrics from {cpu_log}")
        return
    
    print(f"\n🔵 GPU Run: {gpu_log.name}")
    print(f"   Atoms    : {gpu_metrics['n_atoms']:,}")
    print(f"   Steps    : {gpu_metrics['n_steps']:,}")
    print(f"   Timesteps/s: {gpu_metrics['timesteps_per_sec']:.2f}")
    print(f"   Matom-step/s: {gpu_metrics['matom_steps_per_sec']:.3f}")
    
    print(f"\n🟢 CPU Run: {cpu_log.name}")
    print(f"   Atoms    : {cpu_metrics['n_atoms']:,}")
    print(f"   Steps    : {cpu_metrics['n_steps']:,}")
    print(f"   Timesteps/s: {cpu_metrics['timesteps_per_sec']:.2f}")
    print(f"   Matom-step/s: {cpu_metrics['matom_steps_per_sec']:.3f}")
    
    gpu_speed = gpu_metrics['matom_steps_per_sec']
    cpu_speed = cpu_metrics['matom_steps_per_sec']
    
    print(f"\n⚡ SPEEDUP: {gpu_speed / cpu_speed:.2f}x")
    print(f"📈 EFFICIENCY: {gpu_speed / cpu_speed * 0.52:.2f}x (at 52% GPU)")


if __name__ == "__main__":
    gpu_log = Path("gpu_native.log")
    cpu_log = Path("cpu_run_0.log")
    
    if gpu_log.exists() and cpu_log.exists():
        compare_logs(gpu_log, cpu_log)
    else:
        print(f"❌ Missing logs. Available:")
        for log in Path(".").glob("*.log"):
            print(f"   - {log}")
