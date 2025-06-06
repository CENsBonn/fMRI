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

## Install `conda` (Linux)

```shell
❯ conda install -c conda-forge datalad
❯ conda create -n cens
❯ conda activate cens
❯ conda env list

# conda environments:
#
base                   /home/sebelino/miniforge3
cens                 * /home/sebelino/miniforge3/envs/cens

❯ pip install heudiconv
❯ conda install -c conda-forge dcm2niix
```


Download an example dataset:

```shell
❯ wget https://datasets.datalad.org/repronim/heudiconv-reproin-example/reproin_dicom.zip
```

Convert with `heudiconv`:

```shell
heudiconv --files reproin_dicom.zip -f reproin --bids -o bids_datasets
```

The command above should have created a directory structure as follows:
```shell
❯ tree -d
.
└── bids_datasets
    └── Patterson
        └── Coben
            ├── sourcedata
            │   └── sub-001
            │       ├── anat
            │       ├── dwi
            │       ├── fmap
            │       └── func
            └── sub-001
                ├── anat
                ├── dwi
                ├── fmap
                └── func

15 directories
```
