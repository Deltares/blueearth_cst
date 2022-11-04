# docker build from repository root
FROM rocker/r-base:4.2.1

RUN apt-get update -y \
 && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev  \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libnetcdf-dev \
    libudunits2-dev \
    libgdal-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /root/work
WORKDIR /root/work

COPY src/weathergen /root/work/weathergen

RUN Rscript -e "install.packages('devtools',repos = 'http://cran.us.r-project.org')" \
 && Rscript /root/work/weathergen/install_rpackages.r

RUN Rscript -e "install.packages('devtools')"
RUN Rscript -e "install.packages('yaml')"
