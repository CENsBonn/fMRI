---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Set up environment

The following is a step-by-step guide on how to run
[fMRIPrep](https://fmriprep.org) at CENs. It is assumed that you have access to
MRI files in DICOM or BIDS format.

## Create Conda environment

First, open a terminal. If you are using Windows, you should install
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and run all
commands below within WSL.

Check if `conda` is installed:

```console
$ conda info
```

If you get a `command not found` error,
[install Conda](https://conda-forge.org/download/)
and try again.

Next, create a environment and install the required packages:

```console
$ conda create -n cens
$ conda activate cens
$ conda install -c conda-forge dcm2niix
$ conda install -y -c conda-forge datalad
$ conda install -y -c conda-forge bids-validator
$ conda install -y -c conda-forge deno
$ pip install heudiconv
$ pip install fmriprep-docker
```

You should get output similar to the following:

```console
$ conda env list

# conda environments:
#
base                   /home/sebelino/miniforge3
cens                 * /home/sebelino/miniforge3/envs/cens
```

## Clone tools repository

Open a terminal and clone the
[tools](https://github.com/CENsBonn/tools)
repository:
```console
$ git clone https://github.com/CENsBonn/tools
```
Alternatively:
```console
$ git clone git@github.com:CENsBonn/tools
```

If you would like to process data locally, proceed to {doc}`local-setup-bids-conversion`.
If you would like to process data on the HPC cluster, proceed to {doc}`hpc-overview`.
