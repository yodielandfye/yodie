"""
Run GPU-native 500k step simulation.
Uses LAMMPS compute commands for maximum GPU utilization.
"""

from main_gpu_world_emergent_gpu_native import run_lammps_simulation_gpu_native

print("🚀 GPU-NATIVE BIOLOGY TEST - 500k Steps")
print("Parameters: 500k particles, 500k steps, T=1.8, Ea=1.5")
print("⚡ GPU-NATIVE: LAMMPS compute commands + 500k intervals")
print("⚡ CPU sync only once at the end (maximum GPU time)")
print("🚀 Starting simulation...")

run_lammps_simulation_gpu_native(
    n_particles=500_000,  # Scale up for better GPU utilization
    n_steps=500_000,
    reaction_radius=1.5,
    activation_energy_ea=1.5,
    temperature=1.8,
    reaction_check_interval=500_000,  # Check once at the end (GPU runs 99% of time)
)

print("✅ Simulation complete!")

