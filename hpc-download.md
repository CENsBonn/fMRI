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

# Downloading data

At this point, you should have processed your dataset
in the HPC cluster as described in
{doc}`hpc-upload`.
The next step is to download the dataset for further analysis.

## Download the output

Once a job has finished, you can download the output directory to your local
filesystem.
First list the jobs:

```console
$ ./list-jobs.sh
                  JobName               Start    Elapsed      State
------------------------- ------------------- ---------- ----------
 job_20250618_145028_o3Bt 2025-06-18T14:50:30   02:13:03  COMPLETED
```

Download the `output/` directory of the job with the following command:

```console
$ ./download-output.sh job_20250618_145028_o3Bt out
```

The command will download the `output/` directory of the specified job and save
it to the `out/` directory on your machine.
If the job is still in `RUNNING` state, the command above will wait until the
job has finished before downloading the output.

You can also specify `-` as the first argument to select the job from a list
with `fzf`:

```console
$ ./download-output.sh - out
```

Example output directory structure:

```console
$ tree out -d
out
├── logs
├── sourcedata
│   └── freesurfer
│       ├── fsaverage
│       │   ├── label
│       │   ├── mri
│       │   │   ├── orig
│       │   │   └── transforms
│       │   │       └── bak
│       │   ├── mri.2mm
│       │   ├── scripts
│       │   ├── surf
│       │   └── xhemi
│       │       ├── bem
│       │       ├── label
│       │       ├── mri
│       │       │   ├── orig
│       │       │   └── transforms
│       │       │       └── bak
│       │       ├── scripts
│       │       ├── src
│       │       ├── stats
│       │       ├── surf
│       │       ├── tmp
│       │       ├── touch
│       │       └── trash
│       └── sub-001
│           ├── label
│           ├── mri
│           │   ├── orig
│           │   └── transforms
│           │       └── bak
│           ├── scripts
│           ├── stats
│           ├── surf
│           ├── tmp
│           ├── touch
│           └── trash
└── sub-001
    ├── anat
    ├── figures
    └── log
        └── 20250618-122136_a141b6e7-e245-47f7-9900-6c679a2ddef4
```

## Browse dataset

If the dataset is too big to download, you can browse it directly on the HPC
cluster via port-forwarding. Example:

```console
$ ./browse-remote.sh jobs/latest/output

[INFO] Starting HTTP server on marvin:8000...
[INFO] You can now browse http://localhost:8080
```

Now enter http://localhost:8080 in the address bar of your web browser.
