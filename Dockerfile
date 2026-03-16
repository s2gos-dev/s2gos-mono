# ── Stage 1: build ────────────────────────────────────────────
FROM ghcr.io/prefix-dev/pixi:0.50.2-bookworm-slim AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pixi

COPY ./pyproject.toml ./pyproject.toml
COPY ./pixi.lock      ./pixi.lock
COPY ./packages       ./packages

RUN pixi add apache-airflow-providers-cncf-kubernetes

# Everything — including eozilla — is in the lock file
RUN pixi install --locked -e default

# ── Stage 2: runtime ──────────────────────────────────────────
FROM debian:bookworm-slim
WORKDIR /opt/pixi

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/pixi/.pixi/envs/default /opt/pixi/.pixi/envs/default
COPY --from=build /opt/pixi/packages           /opt/pixi/packages
COPY --from=build /opt/pixi/pyproject.toml     /opt/pixi/pyproject.toml
COPY --from=build /opt/pixi/pixi.lock          /opt/pixi/pixi.lock
# Application files — copied directly from build context
COPY ./run_step.py /opt/pixi/run_step.py
COPY ./extra_data  /opt/pixi/hypstar_data
# s2gos_settings.yaml is not baked in — mount it as a ConfigMap in Kubernetes

ENV PIXI_ENV="/opt/pixi/.pixi/envs/default"
ENV PIXI_PROJECT_ROOT="/opt/pixi"
ENV PATH="$PIXI_ENV/bin:$PATH"
ENV LD_LIBRARY_PATH="$PIXI_ENV/lib:$LD_LIBRARY_PATH"
ENV PROJ_DATA="$PIXI_ENV/share/proj"

# Bake eradiate data assets into the image to avoid cold-start at runtime.
# XDG_CACHE_HOME overrides the default ~/.cache so data lands at a fixed,
# user-agnostic path regardless of which UID the pod runs as.
ENV XDG_CACHE_HOME="/opt/pixi/cache"
RUN eradiate data install core gecko monotropa

# Verify the env works
RUN python -c "import s2gos_apps; print('s2gos_apps OK')"

ENTRYPOINT ["python", "/opt/pixi/run_step.py"]
