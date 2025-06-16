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

# Installation

## Setup

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

## fMRIPrep (HPC)

To access the Marvin HPC cluster, you need to [apply for access](https://www.hpc.uni-bonn.de/en/systems/marvin).
Once you have access, you should
generate a SSH keypair,
upload your public key to [FreeIPA](https://freeipa.hpc.uni-bonn.de/),
and connect to the cluster via
SSH/SFTP according to the instructions in the [HPC wiki](https://wiki.hpc.uni-bonn.de/gaining_access).

### Logging in

To generate a keypair:

```console
$ ssh-keygen -t ed25519 -C "sebelin2@uni-bonn.de" -N "" -f ~/.ssh/marvin
```

To log in:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de

        Welcome to the marvin-cluster!
        For more information check our wiki: https://wiki.hpc.uni-bonn.de

[sebelin2_hpc@login02 ~]$ exit
```

### Uploading files

To upload a file or directory (in this example, `bids_datasets`) using `rsync`:
```console
$ rsync -av --info=progress2 -e "ssh -i ~/.ssh/marvin" bids_datasets sebelin2_hpc@marvin.hpc.uni-bonn.de:~
```

To upload a file or directory using SFTP:
```console
$ sftp -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de <<< $'put -R bids_datasets'
```

### Running a sample SLURM job

To run a sample job:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de

[sebelin2_hpc@login02 ~]$ git clone https://github.com/CENsBonn/tools
Cloning into 'tools'...

[sebelin2_hpc@login02 ~]$ cat ./tools/sample-job-script.sh
[sebelin2_hpc@login02 ~]$ sbatch ./tools/sample-job-script.sh
Submitted batch job 22273271
```

The sample job above creates a file named `generated_by_slurm.txt` in your home directory.
The file should appear after a few seconds:

```console
[sebelin2_hpc@login02 ~]$ cat generated_by_slurm.txt
This file was created using a SLURM job
```

In addition, a file will be generated containing the output of the script:

```console
[sebelin2_hpc@login02 ~]$ cat slurm-22273374.out
Running sample job script...
```

### Running a fMRIPrep job

Upload the FreeSurfer license to the HPC cluster:

```console
sftp -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de <<< $'put -R license.txt'
```

Create a `maindir` directory:

```console
[sebelin2_hpc@login02 ~]$ ./tools/setup-fmriprep.sh ./license.txt
```

The script above will create a directory structure as follows:

```console
[sebelin2_hpc@login02 ~]$ find maindir/
maindir/
maindir/work
maindir/fmriprep
maindir/fmriprep/license.txt
maindir/fmriprep/fmriprep-20.2.0.simg
maindir/derivatives
maindir/data
```

Upload the contents of the `bids_datasets` to the `data` directory on the remote:

```console
$ rsync -av --info=progress2 -e "ssh -i ~/.ssh/marvin" bids_datasets/Patterson/Coben/ sebelin2_hpc@marvin.hpc.uni-bonn.de:~/maindir/data
```

On the remote, the contents of the `data` directory should now look like this:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de

[sebelin2_hpc@login01 ~]$ ls maindir/data/
CHANGES			  participants.json  README	 sourcedata  task-rest_bold.json
dataset_description.json  participants.tsv   scans.json  sub-001
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
