# Pixi-based image for the BlueEarth-CST toolbox.
#
# M2 status: rewritten mechanically from the conda+install_rpackages.R
# original. Not yet validated end-to-end on Linux — that work is
# explicitly deferred per dev/tasks/ ("Deferred: Linux replication").
# The image must continue to *parse and build*; full workflow runs
# inside it are exercised when the deferred milestone resumes.

# --- julia binaries ---------------------------------------------------------
# Must satisfy Project.toml's `julia = "~1.11"` compat bound: Wflow.jl v1.0.x
# hangs at JIT under 1.12.x (Wflow.jl#884). Kept equal to
# config/advanced_settings.yml `runtime.julia_version`, which is what the
# workflows select on a juliaup host.
ARG julia_version=1.11.7
FROM julia:${julia_version} AS jul

# --- source files staging ---------------------------------------------------
FROM alpine:latest AS local_files
WORKDIR /root/code
ADD src src
ADD build_model.smk build_model.smk
ADD run_stress_test.smk run_stress_test.smk
ADD analyze_projections.smk analyze_projections.smk

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

# Toolbox revision, BAKED because the image carries no .git -- the sources are
# ADDed individually above, so `git rev-parse` in a container returns nothing
# and every run record would report a null commit with no way to say why.
# `provenance.toolbox_identity()` resolves git first and falls back to this
# file, reporting `commit_source: baked`.
#
# Build with:
#     docker build --build-arg TOOLBOX_COMMIT=$(git rev-parse HEAD) .
#
# An UNSET arg must leave the file ABSENT rather than empty, which is what the
# conditional is for: an empty file would be a commit nobody can look up, while
# an absent one resolves to `commit: null, commit_source: null` -- a record that
# says plainly the revision is unwitnessed instead of one that lies.
ARG TOOLBOX_COMMIT=""
RUN if [ -n "$TOOLBOX_COMMIT" ]; then \
        printf '%s\n' "$TOOLBOX_COMMIT" > /root/work/.toolbox-commit; \
    fi

ENTRYPOINT ["pixi", "run"]
