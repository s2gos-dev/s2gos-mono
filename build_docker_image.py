import shutil
from pathlib import Path
from appligator.airflow.gen_dockerfile import generate

generate(
  package_manager="pixi",
  output_dir=Path("image"),
  extra_files={
      "extra_data": Path("extra_data"),
      # resources/data is not installed by pip (only *.py is included in the
      # package), but s2gos_settings.yaml search_paths expects it at
      # /opt/pixi/packages/s2gos-generator/resources/data at runtime.
      "generator_resources": Path("packages/s2gos-generator/resources/data"),
  },
  build_commands=[],
  packages_dir=Path("packages"),
  local_packages=[
        "s2gos-utils",
        "s2gos-generator",
        "s2gos-simulator",
        "s2gos-apps",
    ],
  runtime_commands=[
      "COPY ./generator_resources /opt/pixi/packages/s2gos-generator/resources/data",
      "COPY ./extra_data /opt/pixi/hypstar_data",
      "RUN XDG_CACHE_HOME=/opt/pixi/cache /opt/pixi/.pixi/envs/default/bin/eradiate data install core gecko monotropa",
      "RUN /opt/pixi/.pixi/envs/default/bin/python -c \"import s2gos_apps; print('s2gos_apps OK')\"",
  ],
)

shutil.copy("pixi.lock", "image/pixi.lock")

dockerfile = Path("image/Dockerfile")
pixi_local = "0.67.2"
content = dockerfile.read_text()
content = content.replace(
    "ghcr.io/prefix-dev/pixi:0.50.2-bookworm-slim",
    f"ghcr.io/prefix-dev/pixi:{pixi_local}-bookworm-slim",
)
content = content.replace(
    "RUN /opt/pixi/.pixi/envs/default/bin/pip install --no-deps --no-build-isolation",
    "RUN /opt/pixi/.pixi/envs/default/bin/python -m ensurepip"
    " && /opt/pixi/.pixi/envs/default/bin/python -m pip install --no-deps",
)
dockerfile.write_text(content)

image_packages = Path("image/packages")
if image_packages.exists():
    shutil.rmtree(image_packages)
shutil.copytree("packages", image_packages)

# run_step.py is hand-maintained at the project root; always overwrite what
# appligator may have placed in image/ so the build context stays in sync.
shutil.copy("run_step.py", "image/run_step.py")
