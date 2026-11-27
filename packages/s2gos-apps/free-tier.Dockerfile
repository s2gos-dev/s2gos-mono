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

# Base image pinned by digest, so that a rebuild gives the same image. To take a newer
# base: `docker pull python:3.12-slim`, put its digest here, regenerate
# free-tier.constraints.txt (see its header) and rebuild.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

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

# Debian security updates published after the pinned base image was built.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Exact versions of every Python package in the image, including transitive ones. The
# `pip install` lines below name the direct dependencies; this file decides the versions.
COPY s2gos-apps/free-tier.constraints.txt /app/constraints.txt

# eozilla from PyPI. All three are pinned to the same version: wraptile requires the
# gavicore/procodile of the same release.
RUN pip install -c /app/constraints.txt \
    "gavicore==0.3.0" \
    "procodile==0.3.0" \
    "wraptile==0.3.0"

# Previous source: the eozilla git archive, used while 0.3.0 was unreleased (it built
# 0.3.0.dev0 from main). Kept for a quick revert, and for testing an unreleased fix:
# uncomment this and comment out the pip install above. All three must come from the
# same ref. Pass a commit SHA as EOZILLA_REF for a reproducible build; the branch
# archive needs no git in the image.
#
# ARG EOZILLA_REF=main
# RUN Z=https://github.com/eo-tools/eozilla/archive/${EOZILLA_REF}.zip \
#     && pip install \
#     "gavicore @ ${Z}#subdirectory=gavicore" \
#     "procodile @ ${Z}#subdirectory=procodile" \
#     "wraptile @ ${Z}#subdirectory=wraptile"

# Runtime dependencies of the free tier, pinned to the versions tested in the pixi dev
# env, except fsspec/s3fs: 2026.6.0 instead of 2026.2.0, for CVE-2026-104851. zarr stays
# below 3: the precalculated stores are zarr v2.
RUN pip install -c /app/constraints.txt \
    "xarray==2024.11.0" "zarr==2.18.7" "numcodecs==0.15.1" \
    "s3fs==2026.6.0" "fsspec==2026.6.0" "universal-pathlib==0.3.10" \
    "shapely==2.1.2" "numpy==1.26.4"

# s2gos-utils with its declared dependencies. s2gos-apps without them: it requires
# s2gos-generator and s2gos-simulator (eradiate), which the free tier never imports.
# `pip check` runs in between: it fails the build if the pinned versions conflict, and
# would always fail after s2gos-apps is installed without its dependencies.
COPY s2gos-utils /src/s2gos-utils
COPY s2gos-apps/pyproject.toml /src/s2gos-apps/pyproject.toml
COPY s2gos-apps/src /src/s2gos-apps/src
RUN pip install -c /app/constraints.txt /src/s2gos-utils \
    && pip check \
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

# Fail the build, not the pod, if the service cannot be imported. The id below is the
# public process id (free_tier.py); keep the two in step — a rename fails here first.
RUN python -c "import sys; \
from s2gos_apps.free_tier.service import service; \
assert list(service.process_registry) == ['observation-simulation-demo'], list(service.process_registry); \
heavy = [m for m in sys.modules if m.startswith(('s2gos_generator', 's2gos_simulator', 'eradiate'))]; \
assert not heavy, heavy"

# pip is only needed at build time. Without it the running container cannot install
# packages, and pip's own vulnerabilities are not in the image.
RUN pip uninstall -y pip

# Numeric user: Kubernetes can only check `runAsNonRoot` against a numeric UID.
RUN useradd --create-home --uid 1000 wraptile
USER 1000:1000

EXPOSE 8008

CMD ["wraptile", "run"]
