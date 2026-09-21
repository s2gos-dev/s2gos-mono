# Wraptile serving only the S2GOS free tier (s2gos_apps.free_tier) on the local backend.
#
# Build with packages/ as the context (packages/.dockerignore keeps it small):
#
#   docker build -f packages/s2gos-apps/free-tier.Dockerfile \
#     -t quay.io/s2gos/free-tier-server:<version> packages
#
# The Helm release (k8s-configs/wraptile, values-free-tier.yaml) overrides the command
# with `wraptile run s2gos_apps.free_tier.service:service` and mounts its own
# settings file. The image also runs on its own:
#
#   docker run -p 8008:8008 --env-file <file with S2GOS_CREDENTIALS__s3ovh__*> \
#     quay.io/s2gos/free-tier-server:<version>

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EOZILLA_SERVER_HOST=0.0.0.0 \
    EOZILLA_SERVER_PORT=8008 \
    EOZILLA_SERVICE="s2gos_apps.free_tier.service:service" \
    SETTINGS_FILE_FOR_DYNACONF=/app/s2gos_settings.yaml \
    AWS_DEFAULT_REGION=de

WORKDIR /app

# eozilla from git until 0.3.0 is on PyPI, then pin "wraptile==0.3.0" "procodile==0.3.0"
# "gavicore==0.3.0" instead. All three come from the same ref: wraptile on main requires
# the gavicore/procodile of main. Pass a commit SHA as EOZILLA_REF for a reproducible
# build. The branch archive needs no git in the image.
ARG EOZILLA_REF=main
RUN Z=https://github.com/eo-tools/eozilla/archive/${EOZILLA_REF}.zip \
    && pip install \
    "gavicore @ ${Z}#subdirectory=gavicore" \
    "procodile @ ${Z}#subdirectory=procodile" \
    "wraptile @ ${Z}#subdirectory=wraptile"

# Runtime dependencies of the free tier, pinned to the versions tested in the pixi dev
# env. zarr stays below 3: the precalculated stores are zarr v2.
RUN pip install \
    "xarray==2024.11.0" "zarr>=2.18,<3" "numcodecs<0.16" \
    "s3fs==2026.2.0" "fsspec==2026.2.0" "universal-pathlib==0.3.10" \
    "shapely==2.1.2" "numpy==1.26.4"

# s2gos-utils with its declared dependencies. s2gos-apps without them: it requires
# s2gos-generator and s2gos-simulator (eradiate), which the free tier never imports.
COPY s2gos-utils /src/s2gos-utils
COPY s2gos-apps/pyproject.toml /src/s2gos-apps/pyproject.toml
COPY s2gos-apps/src /src/s2gos-apps/src
RUN pip install /src/s2gos-utils \
    && pip install --no-deps /src/s2gos-apps \
    && rm -rf /src

# Default settings; SETTINGS_FILE_FOR_DYNACONF points here unless the deployment
# overrides it. s2gos_utils fails to import without a settings file.
RUN printf '%s\n' \
    'common:' \
    '    search_paths: []' \
    '    local_fsspec_cache: "/tmp/fsspec_cache"' \
    '    credential_provider: "environment"' \
    > /app/s2gos_settings.yaml

# Fail the build, not the pod, if the service cannot be imported.
RUN python -c "import sys; \
from s2gos_apps.free_tier.service import service; \
assert list(service.process_registry) == ['free-tier'], list(service.process_registry); \
heavy = [m for m in sys.modules if m.startswith(('s2gos_generator', 's2gos_simulator', 'eradiate'))]; \
assert not heavy, heavy"

RUN useradd --create-home --uid 1000 wraptile
USER wraptile

EXPOSE 8008

CMD ["wraptile", "run"]
