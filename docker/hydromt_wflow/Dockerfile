FROM condaforge/mambaforge:23.1.0-1

COPY environment_python.yaml /tmp/environment.yaml
RUN mamba env update -f /tmp/environment.yaml -n base \
    && mamba clean --all -f -y \
    && pip install hydroengine
RUN pip install git+https://github.com/Deltares/hydromt_wflow.git@main
RUN mamba install -y cartopy
