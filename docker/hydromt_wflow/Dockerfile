FROM condaforge/mambaforge:4.14.0-0

COPY environment_python.yaml /tmp/environment.yaml
RUN mamba env update -f /tmp/environment.yaml -n base \
    && mamba clean --all -f -y \
    && pip install hydroengine
RUN pip install git+https://github.com/Deltares/hydromt_wflow.git@main
RUN mamba install -y cartopy
