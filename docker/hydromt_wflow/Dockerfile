FROM condaforge/mambaforge:4.14.0-0

RUN mamba install -y hydromt_wflow pandas=1.4.1 numpy=1.22.4 seaborn xarray=2022.06.0 \
    && pip install hydroengine
RUN pip install git+https://github.com/Deltares/hydromt_wflow.git@bug/138_datetime
RUN mamba install cartopy