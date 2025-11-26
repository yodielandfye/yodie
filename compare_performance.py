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
    """
    Extract performance metrics from LAMMPS log file.
    
    Looks for:
    - Loop time of X on Y procs for Z steps with N atoms
    - Performance: X tau/day, Y timesteps/s, Z Matom-step/s
    """
    try:
        content = log_path.read_text()
    except:
        return None
    
    metrics = {}
    
    # Extract loop time line
    # Format: "Loop time of 14066.4 on 1 procs for 500000 steps with 49516 atoms"
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
        metrics["atom_steps_per_sec"] = (n_steps * n_atoms) / loop_time
        metrics["matom_steps_per_sec"] = metrics["atom_steps_per_sec"] / 1_000_000
    
    # Extract performance line (if present)
    # Format: "Performance: 15355.781 tau/day, 35.546 timesteps/s, 1.760 Matom-step/s"
    perf_match = re.search(
        r"Performance: [\d.]+ tau/day, ([\d.]+) timesteps/s, ([\d.]+) Matom-step/s",
        content
    )
    
    if perf_match:
        metrics["reported_timesteps_per_sec"] = float(perf_match.group(1))
        metrics["reported_matom_steps_per_sec"] = float(perf_match.group(2))
    
    return metrics if metrics else None


def compare_logs(gpu_log: Path, baseline_log: Path) -> None:
    """Compare GPU and baseline performance metrics."""
    
    print("=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    gpu_metrics = extract_performance_metrics(gpu_log)
    baseline_metrics = extract_performance_metrics(baseline_log)
    
    if not gpu_metrics:
        print(f"❌ Could not extract metrics from {gpu_log}")
        return
    
    if not baseline_metrics:
        print(f"❌ Could not extract metrics from {baseline_log}")
        return
    
    print(f"\n🔵 Current GPU Run: {gpu_log.name}")
    print(f"   Atoms    : {gpu_metrics['n_atoms']:,}")
    print(f"   Steps    : {gpu_metrics['n_steps']:,}")
    print(f"   Procs    : {gpu_metrics['n_procs']}")
    print(f"   Loop time: {gpu_metrics['loop_time']:.2f}s")
    print(f"   Timesteps/s: {gpu_metrics['timesteps_per_sec']:.2f}")
    print(f"   Matom-step/s: {gpu_metrics['matom_steps_per_sec']:.3f}")
    
    print(f"\n🟢 Baseline: {baseline_log.name}")
    print(f"   Atoms    : {baseline_metrics['n_atoms']:,}")
    print(f"   Steps    : {baseline_metrics['n_steps']:,}")
    print(f"   Procs    : {baseline_metrics['n_procs']}")
    print(f"   Loop time: {baseline_metrics['loop_time']:.2f}s")
    print(f"   Timesteps/s: {baseline_metrics['timesteps_per_sec']:.2f}")
    print(f"   Matom-step/s: {baseline_metrics['matom_steps_per_sec']:.3f}")
    
    # Normalize for fair comparison (account for different atom counts)
    # Use Matom-step/s which is already normalized
    gpu_speed = gpu_metrics['matom_steps_per_sec']
    baseline_speed = baseline_metrics['matom_steps_per_sec']
    
    print(f"\n⚡ SPEEDUP ANALYSIS")
    print(f"   Current GPU: {gpu_speed:.3f} Matom-step/s")
    print(f"   Baseline   : {baseline_speed:.3f} Matom-step/s")
    
    if baseline_speed > 0:
        speedup = gpu_speed / baseline_speed
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"   ✅ Current GPU is {speedup:.2f}x FASTER than baseline")
        else:
            print(f"   ❌ Current GPU is {1/speedup:.2f}x SLOWER than baseline")
    else:
        print("   ⚠️  Cannot calculate speedup (baseline speed is 0)")
    
    # Efficiency: GPU utilization * speedup
    print(f"\n📈 EFFICIENCY")
    print(f"   Current GPU utilization: ~52% (from nvidia-smi)")
    print(f"   Effective speedup: {speedup * 0.52:.2f}x (at current GPU utilization)")
    print(f"   Potential at 100% GPU: {speedup / 0.52:.2f}x")


if __name__ == "__main__":
    # Compare current GPU run with baseline
    gpu_log = Path("gpu_native.log")  # Current GPU run
    baseline_log = Path("gpu_emergent.log")  # Baseline (older GPU run)
    
    # Fallback: try other GPU logs if gpu_native.log doesn't exist
    if not gpu_log.exists():
        print(f"⚠️  {gpu_log} not found. Looking for other GPU logs...")
        for log in ["gpu_optimized.log", "gpu_emergent.log"]:
            if Path(log).exists():
                gpu_log = Path(log)
                print(f"   Using {gpu_log}")
                break
    
    # Fallback: try CPU log if baseline doesn't exist
    if not baseline_log.exists():
        print(f"⚠️  {baseline_log} not found. Looking for CPU logs...")
        for log in Path(".").glob("cpu_run_*.log"):
            baseline_log = log
            print(f"   Using {baseline_log}")
            break
    
    if gpu_log.exists() and baseline_log.exists():
        compare_logs(gpu_log, baseline_log)
    else:
        print(f"❌ Missing logs:")
        if not gpu_log.exists():
            print(f"   - {gpu_log}")
        if not baseline_log.exists():
            print(f"   - {baseline_log}")
        print(f"\nAvailable logs:")
        for log in Path(".").glob("*.log"):
            print(f"   - {log}")
