"""
Helper entrypoint that wires the GPU-native emergent substrate into a single
command.  Use this to launch controlled test runs without editing the main
module directly.
"""

from main_gpu_world_emergent_gpu_native import (
    EmergentConfig,
    run_lammps_simulation_gpu_native,
)


if __name__ == "__main__":
    # Fast feedback loop: keep particle count high, but only integrate 5k steps.
    cfg = EmergentConfig(n_steps=5_000)
    print("🚀 GPU-NATIVE BIOLOGY TEST - 5k Steps")
    print("Parameters: 500k particles, 5k steps, T=1.8, Ea=1.5")
    print("⚡ GPU-NATIVE: LAMMPS compute commands, single CPU sync at the end")
    run_lammps_simulation_gpu_native(cfg)
    print("✅ Short-run simulation complete! Check gpu_native.log for loop stats.")

