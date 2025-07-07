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

# Uploading data

Once you are able to connect to the HPC cluster,
you will want to upload your dataset to it.
The next steps will depend on your use case.

* Is your dataset DICOM data?
  * If yes, is the dataset too big or computationally expensive to be processed further on your local machine?
    * If yes, see: {ref}`id-upload-dicom-dataset`.
    * If no, {doc}`run BIDS conversion locally<local-setup-bids-conversion>`.
* Is your dataset BIDS data?
  * If yes, see: {ref}`id-upload-bids-dataset`.

(id-upload-dicom-dataset)=
## Upload DICOM dataset

Let us assume that you have access to an external hard drive containing a
directory with DICOM files. If the directory is very large (100+ GB), you may
want to upload it to the HPC cluster before processing it further.

Mount the external hard drive and find the directory containing the DICOM
files:

```console
$ sudo mount /dev/sda2 /mnt
$ ls /mnt/
```

Let us assume that the directory `/mnt/path/to/dicom_dir/` contains
subdirectories named `sub-102`, `sub-103`, etc. Each subdirectory in turn
contains `.dcm` files. Upload this directory to the HPC cluster using `rsync`
like so:

```console
$ ./upload-input.sh /mnt/path/to/dicom_dir/ reproin-dicom 60
```

The command above creates a new
[workspace](https://wiki.hpc.uni-bonn.de/en/marvin/workspaces) on the HPC
cluster with the name `reproin-dicom` and an expiry time of `60` days.
The directory is then uploaded to this workspace directory.
You can confirm that the upload was successful by finding the workspace
directory and listing files within it:

```console
$ ssh marvin ws_list
$ ssh marvin ls /lustre/scratch/data/sebelin2_hpc-reproin-dicom/
$ ./list-ws.sh
```

Now that the DICOM dataset is uploaded, the next step is to
{ref}`convert it <id-convert-dicom-to-bids>`
to BIDS format.

(id-upload-bids-dataset)=
## Upload BIDS dataset

The following command can be used to upload a BIDS converted directory to the
remote server. The directory to be uploaded must contain a
`dataset_description.json` file. In the example below, a new
[workspace](https://wiki.hpc.uni-bonn.de/en/marvin/workspaces)
will be created with the name `reproin` and expiry time `7` days, and the
directory will be uploaded to that workspace.

```console
$ ./upload-input-bids.sh ../fMRI/example/bids_datasets/Patterson/Coben reproin 7
```

A new workspace is now created and contains the uploaded files:

```console
$ ssh marvin ws_list
id: reproin
     workspace directory  : /lustre/scratch/data/sebelin2_hpc-reproin
     remaining time       : 6 days 23 hours
     creation time        : Tue Jun 17 14:12:11 2025
     expiration date      : Tue Jun 24 14:12:11 2025
     filesystem name      : scratch
     available extensions : 3

$ ssh marvin ls /lustre/scratch/data/sebelin2_hpc-reproin/
CHANGES
dataset_description.json
participants.json
participants.tsv
README
scans.json
sourcedata
sub-001
task-rest_bold.json
```

Alternatively, you can create a workspace manually using `ws_allocate` and
upload files manually using `rsync` or `sftp`.
