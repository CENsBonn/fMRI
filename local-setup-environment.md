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
install Miniconda and try again.
If you are using WSL, you can do so by running:

```console
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
$ bash miniconda.sh -b -p "$HOME/miniconda"
$ rm -f miniconda.sh
$ eval "$($HOME/miniconda/bin/conda shell.bash hook)"
$ conda init
```
Then restart the terminal.

Next, create a environment and install the required packages:

```console
$ conda create -y -n cens
$ conda activate cens
$ conda install -y -c conda-forge dcm2niix
$ conda install -y -c conda-forge datalad
$ conda install -y -c conda-forge bids-validator
$ conda install -y -c conda-forge deno
$ conda install -y -c conda-forge tree
$ pip install heudiconv
$ pip install fmriprep-docker
```

Before proceeding, make sure you have activated your Conda environment:

```console
$ conda activate cens
```

You will need to run the command above every time you open a new terminal.

(id-clone-tools-repository)=
## Clone tools repository

Clone the
[tools](https://github.com/CENsBonn/tools)
repository:
```console
$ cd $HOME
$ git clone https://github.com/CENsBonn/tools
```
Alternatively:
```console
$ cd $HOME
$ git clone git@github.com:CENsBonn/tools
```

## Install Docker

If you are planning to run fMRIPrep locally, you should install
[Docker](https://docs.docker.com/get-started/get-docker/). If you are using
WSL, download Docker Desktop for Windows and activate the WSL integration in
the Docker settings. You may need to restart your WSL terminal afterwards.

## Next steps

If you would like to process fMRI data locally, proceed to {doc}`local-setup-bids-conversion`.

If you would like to process data on the HPC cluster, jump ahead to the
{doc}`HPC guide<hpc-overview>`.
