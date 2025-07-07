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

# BIDS conversion

In this guide, we will make use of an example DICOM dataset named `reproin`
and convert it to BIDS format.
If your dataset is already in BIDS format, you may skip this page.

Create a directory and enter it:

```console
$ cd $HOME
$ mkdir -p dicom-example
$ cd dicom-example
```

Download the dataset by running:

```console
$ wget https://datasets.datalad.org/repronim/heudiconv-reproin-example/reproin_dicom.zip
```

```console
$ tree
.
└── reproin_dicom.zip

1 directory, 1 file
```

Convert it to BIDS with
[HeuDiConv](https://heudiconv.readthedocs.io/en/latest/):

```console
$ conda activate cens
$ heudiconv --files reproin_dicom.zip -f reproin --bids -o bids_datasets
```

In the command above, the `--files` option accepts a zip file
(`reproin_dicom.zip`), but it is also possible to
supply a path to a directory containing the DICOM files.

The command above should have created a directory structure as follows:
```console
$ tree -d
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

In this case, the `Coben` directory contains the BIDS converted data, because
it contains the `dataset_description.json` file:

```console
$ find bids_datasets/ -name dataset_description.json
bids_datasets/Patterson/Coben/dataset_description.json
```

If desired, validate the dataset with `bids-validator`:

```console
$ deno run -ERWN jsr:@bids/validator bids_datasets/Patterson/Coben/
```
