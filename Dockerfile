FROM quay.io/jupyterhub/singleuser

USER root

#install LDAP stuff
RUN apt-get -y update && apt-get -y install libnss-ldap libpam-ldap ldap-utils nscd
# COPY ldap.conf /etc/ldap.conf
# COPY nsswitch.conf /etc/nsswitch.conf
# COPY start-singleuser.sh /usr/local/bin/
# RUN chmod 755 /usr/local/bin/start-singleuser.sh #ensure execute permissions are set

#install stuff that might be useful to users
RUN apt-get -y install git mc nano vim-tiny curl zip ncdu bwm-ng htop

#install extra oceanography science packages
#and the JupyterLab NVIDIA Dashboard extension
RUN conda install -y -c conda-forge cuda-nvcc cuda-nvrtc "cuda>=12.0" parallel graphviz && \
pip install --upgrade pip && \
pip install numpy scipy matplotlib pandas xarray dask[complete] netCDF4 h5netcdf bottleneck pint cftime zarr fsspec intake intake-xarray  \
pooch s3fs hvplot cartopy seaborn scikit-learn scikit-image tensorflow torch torchvision pytorch-lightning numba cupy-cuda12x gcsfs graphviz  \
jupyterlab-nvdashboard jupyter-resource-usage

RUN apt-get -y install man-db && yes y | unminimize

#clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*