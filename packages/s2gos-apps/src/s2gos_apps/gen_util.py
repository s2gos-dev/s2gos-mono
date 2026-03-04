from s2gos_generator import SceneGenConfig, SceneGenerationPipeline


def generation_from_config(config: SceneGenConfig):
    """3D scene generation from a scene configuration."""

    print("\n")
    print("=" * 60)
    print("Generating scene description...")

    # Step 1: Create and validate configuration
    print("Configuration validation")
    if not config:
        return None

    # Display configuration summary
    print("\nConfiguration Summary:")
    print(f"  Scene: {config.scene_name}")
    print(
        f"  Location: {config.location.center_lat:.4f}°, {config.location.center_lon:.4f}°"
    )
    print(f"  AOI: {config.location.aoi_size_km} km²")
    print(f"  Resolution: {config.target_resolution_m} m")

    print("Generating scene with configuration")

    try:
        pipeline = SceneGenerationPipeline(config)

        # Generate DAG visualization
        print("\nGenerating DAG visualization")
        dag_path = pipeline.visualize_dag()
        if dag_path:
            print(f"DAG saved to: {dag_path}")

        # Print execution schedule
        print("\nResource execution order:")
        dependencies = pipeline.get_resource_dependencies()
        for resource_id, deps in dependencies.items():
            print(f"  {resource_id} (depends on: {deps or 'none'})")

        pipeline.run()

        print("Scene generated successfully!")
        print(f" Location: {config.location.center_lat}, {config.location.center_lon}")
        print(
            f" Target: {config.location.aoi_size_km}km² at {config.target_resolution_m}m"
        )
        if config.buffer is not None:
            print(
                f"  Buffer: {config.buffer.size_km}km at {config.buffer.resolution_m}m"
            )
        if config.background is not None:
            print(
                f"  Background: {config.background.size_km}km at {config.background.resolution_m}m"
            )
        print(f"  Output: {config.scene_output_dir}")
    except Exception:
        raise

    # Summary
    print("\n" + "=" * 60)
    print(f"Output directory: {config.scene_output_dir}")

    return config.scene_output_dir.upath / f"{config.scene_name}.yml"
