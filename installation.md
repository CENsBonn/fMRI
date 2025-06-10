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

## Installation

First, open a terminal and check if `conda` is installed:

```console
$ conda info
```

If you get a `command not found` error, [install Conda](https://conda-forge.org/download/).

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

```console
$ conda env list

# conda environments:
#
base                   /home/sebelino/miniforge3
cens                 * /home/sebelino/miniforge3/envs/cens
```

## BIDS conversion

Download an example dataset:

```console
$ wget https://datasets.datalad.org/repronim/heudiconv-reproin-example/reproin_dicom.zip
```

Convert with `heudiconv`:

```console
$ heudiconv --files reproin_dicom.zip -f reproin --bids -o bids_datasets
```

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

Validate the dataset with `bids-validator`:

```console
$ deno run -ERWN jsr:@bids/validator bids_datasets/Patterson/Coben/
```

## fMRIPrep

Before running `fMRIPrep`, you need to obtain a license file for FreeSurfer.
Fill out the form on the following page to obtain a
`license.txt` file:

https://surfer.nmr.mgh.harvard.edu/registration.html

Some of the fields can be left blank. Make sure to set:

* **E-mail address**: (Your e-mail address)
* **Type of institution which will use this software**: `Non-profit education/research`
* **Operating System/Platform**: `Linux/Intel`
* **Number of Users**: `1`

After submitting the form, you will receive an e-mail with `license.txt` attached.

Download the license file and copy it to the current working directory:

```
$ cp ~/Downloads/license.txt .
```

Ensure you have installed [Docker](https://docs.docker.com/get-started/get-docker/),
then run `fMRIPrep` on the dataset:

```console
$ fmriprep-docker bids_datasets/Patterson/Coben fmriprep_output participant \
    --participant_label sub-001 \
    --write-graph \
    --fs-no-reconall \
    --notrack \
    --fs-license-file license.txt \
    --work-dir fmriprep_tmp
```

The command takes approximately 30 minutes to finish. The majority of the time
will likely be spent on the nipype registration step:

```
250610-08:25:38,253 nipype.workflow INFO:
	 [Node] Finished "registration", elapsed time 1116.977937s.
```

Once the command finishes, you should have a directory structure as follows:

```console
$ tree -d -L1
.
├── bids_datasets
├── fmriprep_output
└── fmriprep_tmp
```

The `fmriprep_tmp` directory can be safely removed.

The `fmriprep_output` directory contains the preprocessed data which can be
used for further analysis.
To look at the results, open `sub-001.html` in a browser:

```console
$ chromium fmriprep_output/sub-001.html
```

## Troubleshooting

### Command not found

If you get an error similar to the following:

```console
zsh: command not found: fmriprep-docker
```

then make sure that the `cens` conda environment is activated:

```console
$ conda activate cens
```

If the above command fails, make sure you have followed the [installation instructions](#installation).
