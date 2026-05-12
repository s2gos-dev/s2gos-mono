import shutil
from pathlib import Path
from appligator.airflow.gen_dockerfile import generate

generate(
  package_manager="pixi",
  output_dir=Path("image"),
  extra_files={"extra_data": Path("extra_data")},
  build_commands=[],
  packages_dir = Path("packages"),
  local_packages=[
        "s2gos-utils",
        "s2gos-generator",
        "s2gos-simulator",
        "s2gos-apps",
    ],
   # runtime_commands=[...],
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

# # build_image.py
# from pathlib import Path
# from textwrap import dedent
#
# from procodile import WorkflowStepRegistry
# from appligator.airflow.gen_image import gen_image
#
# registry = WorkflowStepRegistry()
#
# gen_image(
#     registry,
#     image_name="myrepo/s2gos-apps:latest",
#     extra_files={"extra_data": Path("extra_data")},
#     build_stage_extra=dedent("""\
#           RUN pixi add apache-airflow-providers-cncf-kubernetes
#
#           # Everything — including eozilla — is in the lock file"""),
#     dockerfile_extra=dedent("""\
#           COPY ./extra_data  /opt/pixi/hypstar_data
#           # s2gos_settings.yaml is not baked in — mount it as a ConfigMap in Kubernetes
#
#           RUN eradiate data install core gecko monotropa
#           RUN python -c "import s2gos_apps; print('s2gos_apps OK')" """),
# )