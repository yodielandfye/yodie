"""
Run GPU-optimized 500k step simulation.
"""

from main_gpu_world_emergent_gpu_opt import run_lammps_simulation_gpu_optimized

print("🚀 GPU-OPTIMIZED BIOLOGY TEST - 500k Steps")
print("Parameters: 50k particles, 500k steps, T=1.8, Ea=1.5")
print("⚡ Full GPU acceleration with periodic bond creation")
print("🚀 Starting simulation...")

run_lammps_simulation_gpu_optimized(
    n_particles=50_000,
    n_steps=500_000,
    reaction_radius=1.5,
    activation_energy_ea=1.5,
    temperature=1.8,
    reaction_check_interval=100,  # Check for reactions every 100 steps
)

print("✅ Simulation complete!")
