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

### Log in

To generate a keypair:

```console
$ ssh-keygen -t ed25519 -N "" -f ~/.ssh/marvin
```

Make sure you are able to log in, then exit:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de

        Welcome to the marvin-cluster!
        For more information check our wiki: https://wiki.hpc.uni-bonn.de

[sebelin2_hpc@login02 ~]$ exit
```

Add an entry to your local `~/.ssh/config`:

```console
$ git clone https://github.com/CENsBonn/tools
$ cd tools/
$ ./configure-remote-connection.sh sebelin2_hpc ~/.ssh/marvin
```

The command above allows you to log in without having to specify your username
or SSH key:

```console
$ ssh marvin

        Welcome to the marvin-cluster!
        For more information check our wiki: https://wiki.hpc.uni-bonn.de

[sebelin2_hpc@login02 ~]$ exit
```

### Set up fMRIPrep

Upload the FreeSurfer license to the HPC cluster:

```console
$ ./upload-license.sh ../fMRI/example/license.txt
```

Build the fMRIPrep Singularity image on the HPC cluster:

```console
$ ./build-fmriprep-image.sh
```

After running the commands above, a directory containing two new files should
have appeared in your home directory:

```console
$ ssh marvin ls fmriprep/
fmriprep-25.1.1.simg
license.txt
```

### Upload dataset

The following command can be used to upload a BIDS converted directory to the
remote server. The directory must contain a `dataset_description.json` file. In
the example below, a new
[workspace](https://wiki.hpc.uni-bonn.de/en/marvin/workspaces)
will be created with the name `reproin` and expiry time `7` days, and the
directory will be uploaded to that workspace.

```console
$ ./upload-input.sh ../fMRI/example/bids_datasets/Patterson/Coben reproin 7
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

### Run a sample SLURM job

There is an example SLURM job script in the `tools` directory
which you can use for testing.
To run the job:

```console
$ ./run-job.sh reproin sample-job-script.slurm extra_argument another_argument

Created job directory: jobs/job_20250618_113525_zodD
sending incremental file list
sample-job-script.slurm
            275 100%    0.00kB/s    0:00:00 (xfr#1, to-chk=0/1)

sent 405 bytes  received 35 bytes  880.00 bytes/sec
total size is 275  speedup is 0.62
Info: creating workspace.
/lustre/scratch/data/sebelin2_hpc-reproin_job_20250618_113525_zodD_out
remaining extensions  : 3
remaining time in days: 7
Info: creating workspace.
/lustre/scratch/data/sebelin2_hpc-reproin_job_20250618_113525_zodD_work
remaining extensions  : 3
remaining time in days: 1
Submitted batch job 22282522
```

The command above submits a batch job based on the `sample-job-script.slurm`
SLURM script. The first argument (`reproin`) is the name of the workspace
containing the previously uploaded dataset.
The extra arguments (`extra_argument`, `another_argument`) are optional
and will be passed to the SLURM script.

In the example above, the results of the job will be stored in the directory
`~/jobs/job_20250618_113525_zodD` on the remote.
For convenience, you can use the symbolic link `~/jobs/latest` to access this
directory.
Once the job has completed (after a few seconds),
the directory will contain the following files:

```console
$ ssh marvin ls jobs/latest/
input
output
sample-job-script.slurm
slurm.out
work
```

`input` is a symbolic link to the workspace directory containing the uploaded
files:

```console
$ ssh marvin ls jobs/latest/input
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

The printed output of the job is stored in `slurm.out`.
To inspect it:

```console
$ ssh marvin cat jobs/latest/slurm.out
Running sample job script with arguments: extra_argument another_argument
Input directory: input
Output directory: output
Work directory: work
```

The job script (`sample-job-script.slurm`) is programmed to
write an example file to the output directory. To inspect it:

```console
$ ssh marvin ls jobs/latest/output
generated_by_sample_job.txt

$ ssh marvin cat jobs/latest/output/generated_by_sample_job.txt
This output was generated by the sample SLURM job
```

### Run an fMRIPrep job

To run `fMRIPrep` on the uploaded dataset, you can use the following command:

```console
$ ./run-job.sh reproin fmriprep.slurm sub-001
```

The extra argument `sub-001` above is the label of the participant you would
like to process.

If you'd like to receive an email notification when the job is finishes, you
can edit `fmriprep.slurm` and add a `--mail-user` option before running it:

```diff
diff --git a/fmriprep.slurm b/fmriprep.slurm
index 02a899b..93e14b7 100755
--- a/fmriprep.slurm
+++ b/fmriprep.slurm
@@ -5,6 +5,7 @@
 #SBATCH --cpus-per-task=32
 #SBATCH --mem=32000
 #SBATCH --mail-type=END,FAIL,TIME_LIMIT
+#SBATCH --mail-user=sebelin2@uni-bonn.de
 #SBATCH --time=03:00:00

 set -Eeuo pipefail
```

To monitor the job, you can `tail` the printed output. Th output will then be
printed to your terminal continuously:

```console
$ ./tail-output.sh
250618-12:22:11,496 nipype.workflow INFO:
	 [Node] Executing "_denoise0" <nipype.interfaces.ants.segmentation.DenoiseImage>
250618-12:22:15,738 nipype.workflow INFO:
	 [Node] Finished "lap_tmpl", elapsed time 7.441261s.
250618-12:22:17,377 nipype.workflow INFO:
	 [Node] Setting-up "fmriprep_25_1_wf.sub_001_wf.anat_fit_wf.brain_extraction_wf.mrg_tmpl" in "/work/fmriprep_25_1_wf/sub_001_wf/anat_fit_wf/brain_extraction_wf/mrg_tmpl".
250618-12:22:17,398 nipype.workflow INFO:
	 [Node] Executing "mrg_tmpl" <nipype.interfaces.utility.base.Merge>
250618-12:22:17,399 nipype.workflow INFO:
	 [Node] Finished "mrg_tmpl", elapsed time 0.000148s.
```

To stop printing, press CTRL+C. The job will continue running.

During the execution of the job, the `output/` and `work/` directories
will be populated with files. You can confirm this like so:

```console
$ ssh marvin "ls jobs/latest/output"
logs
sourcedata
sub-001

$ ssh marvin "ls jobs/latest/work"
20250618-122136_a141b6e7-e245-47f7-9900-6c679a2ddef4
fmriprep_25_1_wf
```

### Download the output

Once the job has finished, you can download the output directory to your local
filesystem.
First list the running jobs:

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
