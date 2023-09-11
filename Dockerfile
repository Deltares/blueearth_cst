# julia intermediate
ARG julia_version=1.8.2
FROM julia:${julia_version} as jul

FROM alpine:latest AS local_files

WORKDIR /root/code
ADD src src
ADD Snakefile_model_creation Snakefile_model_creation
ADD Snakefile_climate_experiment Snakefile_climate_experiment
ADD Snakefile_climate_projections Snakefile_climate_projections

# Python env
FROM condaforge/mambaforge:4.14.0-0

WORKDIR /root/work

COPY environment.yml environment.yaml
RUN mamba env create -f environment.yaml -n snakemake -q \
    && . activate snakemake \
    && conda clean --all -y \
    && echo "source activate snakemake" > ~/.bashrc \
    && rm environment.yaml
ENV PATH /opt/conda/envs/snakemake/bin:${PATH}

# Julia wflow
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"
RUN apt-get update -y \
 && apt-get install -y \
    build-essential libatomic1 gfortran perl wget m4 cmake pkg-config curl \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=jul /usr/local/julia /opt/julia

ENV PATH /opt/julia/bin:${PATH}

RUN julia -e "import Pkg; Pkg.add(\"Wflow\"); using Wflow;"

COPY src/weathergen/install_rpackages.R /tmp/install_rpackages.R

RUN Rscript -e "install.packages('devtools',repos = 'http://cran.us.r-project.org')"
RUN Rscript /tmp/install_rpackages.R

COPY --from=local_files /root/code /root/work/
