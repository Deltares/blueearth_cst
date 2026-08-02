# Pixi-based image for the BlueEarth-CST toolbox.
#
# M2 status: rewritten mechanically from the conda+install_rpackages.R
# original. Not yet validated end-to-end on Linux — that work is
# explicitly deferred per dev/followups.md ("Deferred: Linux replication").
# The image must continue to *parse and build*; full workflow runs
# inside it are exercised when the deferred milestone resumes.

# --- julia binaries ---------------------------------------------------------
ARG julia_version=1.12.6
FROM julia:${julia_version} AS jul

# --- source files staging ---------------------------------------------------
FROM alpine:latest AS local_files
WORKDIR /root/code
ADD src src
ADD Snakefile_model_creation Snakefile_model_creation
ADD Snakefile_climate_experiment Snakefile_climate_experiment
ADD Snakefile_climate_projections Snakefile_climate_projections

# --- pixi-managed Python + R env --------------------------------------------
FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /root/work

# OS deps for Julia / Wflow build artefacts
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential libatomic1 gfortran perl wget m4 cmake pkg-config curl git \
 && rm -rf /var/lib/apt/lists/*

# Bring in Julia binaries from layer 1
COPY --from=jul /usr/local/julia /opt/julia
ENV PATH=/opt/julia/bin:${PATH}

# Pixi env declaration + lock + Julia project lock
COPY pixi.toml ./
COPY Project.toml Manifest.toml ./

# Scripts the install task needs (weathergenr installer)
COPY dev/scripts dev/scripts

# Native conda-forge deps (Python + R toolchain)
RUN pixi install

# weathergenr (R) + Julia/Wflow (Pkg.instantiate)
RUN pixi run install

# Workflow source code
COPY --from=local_files /root/code /root/work/

ENTRYPOINT ["pixi", "run"]
