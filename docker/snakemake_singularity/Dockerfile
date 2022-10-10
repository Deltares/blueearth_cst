FROM condaforge/mambaforge

WORKDIR /root/work
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup \
    debootstrap
# Install go
ENV HOME="/root"
ENV GO_VERSION=1.17 OS=linux ARCH=amd64
RUN wget https://dl.google.com/go/go${GO_VERSION}.${OS}-${ARCH}.tar.gz \
 && tar -C ${HOME} -xzvf go${GO_VERSION}.${OS}-${ARCH}.tar.gz \
 && rm go${GO_VERSION}.${OS}-${ARCH}.tar.gz

ENV GOPATH=${HOME}/go
ENV GOBIN=${GOPATH}/bin
ENV PATH=${GOPATH}/bin:${PATH}

# Install Singularity
ENV SINGULARITY_VERSION=3.5.3
RUN wget https://github.com/singularityware/singularity/releases/download/v${SINGULARITY_VERSION}/singularity-${SINGULARITY_VERSION}.tar.gz \
 && tar -xzvf singularity-${SINGULARITY_VERSION}.tar.gz \
 && rm singularity-${SINGULARITY_VERSION}.tar.gz \
 && cd singularity \
 && ./mconfig \
 && cd builddir \
 && make \
 && make install \
 && rm -r ${HOME}/work/singularity

# Install Snakemake
COPY environment.yaml environment.yaml
RUN mamba env create -f environment.yaml -n snakemake -q \
 && . activate snakemake \
 && conda clean --all -y \
 && echo "source activate snakemake" > ~/.bashrc
ENV PATH /opt/conda/envs/snakemake/bin:${PATH}
