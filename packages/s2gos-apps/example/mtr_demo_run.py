"""
Unified Patagonia MTR Demo Simulation - Seasonal Support
Runs both December (summer) and June (winter) with generation + simulation.

Updated to use simplified mtr_demo processors.
"""

from s2gos_apps.processes.pnp_gui import (
    Month,
    ObservationType,
    pnp_gui_generation,
    pnp_gui_simulation,
)

# Sensor observations to include
observations = [
    # ObservationType.HYPSTAR,
    # Uncomment to add more:
    # ObservationType.CHIME,
    # ObservationType.MSI,
    # ObservationType.HYPSTAR,
    ObservationType.RGB_CAMERA,
    # ObservationType.SATELLITE_PIXEL_HDRF,
]

# Seasons to run
seasons = [Month.JUNE]

# Scene parameters
random_seed = 13
gmt_hour = 17
spp = 4  # samples per pixel

for month in seasons:
    print(f"\n=== Running season: {month.value.upper()} ===")

    # Season-specific output base directory
    scene_name = f"test_rgb_{month.value}"

    ## Generate Scene
    pnp_gui_generation(month=month, random_seed=random_seed, scene_name=scene_name)

    for observation in observations:
        print(f"\n  Running {observation} simulation...")

        simulation_output_dir = pnp_gui_simulation(
            scene_name=scene_name,
            month=month,
            hour_utc=gmt_hour,
            observation=observation,
            spp=spp,
            sim_name=f"test_{observation}_{month.value}",
        )
        print(f"{simulation_output_dir=}")
        print(type(simulation_output_dir))

        if simulation_output_dir:
            print(f"  {observation} complete: {simulation_output_dir}")

    print(f"\nFinished {month.value}")
