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

# Processing data

At this point, you should have uploaded your dataset
to the HPC cluster as described in
[HPC: Uploading data](hpc-upload).
The next step is to process the dataset using SLURM jobs.

## Run a sample SLURM job

There is an
example SLURM job script
([`example/job.slurm`](https://github.com/CENsBonn/tools/blob/main/job/example/job.slurm))
which you can use to test running a SLURM job.
The script runs quickly and simply counts the number of files in your dataset.
Run the job like so:

```console
$ cd tools/
$ ./run-job.sh reproin-dicom ./job/example -- extra_argument another_argument

Created job directory: jobs/job_20250618_113525_zodD
sending incremental file list
job.slurm
            275 100%    0.00kB/s    0:00:00 (xfr#1, to-chk=0/1)

sent 405 bytes  received 35 bytes  880.00 bytes/sec
total size is 275  speedup is 0.62
Info: creating workspace.
/lustre/scratch/data/sebelin2_hpc-reproin-dicom_job_20250618_113525_zodD_out
remaining extensions  : 3
remaining time in days: 7
Info: creating workspace.
/lustre/scratch/data/sebelin2_hpc-reproin-dicom_job_20250618_113525_zodD_work
remaining extensions  : 3
remaining time in days: 1
Submitted batch job 22282522
```

The command above submits a batch job based on the `example/job.slurm`
SLURM script.

* The first argument (`reproin-dicom`) is the name of the workspace containing
the dataset you uploaded earlier.
* The second argument (`./job/example`) is the directory containing the `job.slurm` script.
* The extra arguments (`extra_argument`, `another_argument`) are optional and
will be passed to the SLURM script.

In the example above, the results of the job will be stored in the directory
`~/jobs/job_20250618_113525_zodD` on the remote.
To list the files in this directory, you can do:

```console
$ ssh marvin ls jobs/job_20250618_113525_zodD
```

For convenience, you can alternatively use the symbolic link `~/jobs/latest` to access
this directory.
Once the job has completed (after a few seconds),
the job directory will contain the following files:

```console
$ ssh marvin ls jobs/latest/
input
output
scripts
slurm.out
work
```

`input` is a symbolic link to the workspace directory containing the
dataset you uploaded earlier:

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

`slurm.out` contains the printed messages of the job.
To inspect it:

```console
$ ssh marvin cat jobs/latest/slurm.out
Running sample job script with arguments: extra_argument another_argument
Input directory: input
Output directory: output
Work directory: work
```

`output` is a symbolic link to a workspace where the output of the job is
stored.
In this case,
the job script `example/job.slurm` is programmed to
write an example file (`generated_by_sample_job.txt`)
to the output directory. To inspect it:

```console
$ ssh marvin ls jobs/latest/output
generated_by_sample_job.txt

$ ssh marvin cat jobs/latest/output/generated_by_sample_job.txt
The /lustre/scratch/data/sebelin2_hpc-reproin-dicom directory contains 791 files.
```

`work` is a symlink to a workspace where temporary files may be written by the
job. In this example, the SLURM script does not touch it at all.

## Monitor output of job

While running a long-running job, you may want to monitor its progress in
real-time. You can monitor the contents of the file `jobs/latest/slurm.out` by
running:

```console
$ ./tail-output.sh

Running sample job script with arguments: extra_argument another_argument
Input directory: input
Output directory: output
Work directory: work
```

The command above will continuously print the output of the job to your
terminal while the job is running. Press `CTRL+C` to stop.

If you want to monitor a specific file, pass it as an argument like so:

```console
$ ./tail-output.sh jobs/job_20250625_145225_Y32s/slurm.out
```

## List jobs

To get a list of SLURM jobs:

```console
$ ./list-jobs.sh

                  JobName               Start    Elapsed      State
------------------------- ------------------- ---------- ----------
 job_20250618_132739_zodD 2025-06-18T13:27:42   00:00:10  COMPLETED
```

## List workspaces

Running a job using `run-job.sh`
causes two new workspaces to be created automatically:
one for the `output` directory and one for the `work` directory.
You can view a list of existing workspaces by running:

```console
$ ./list-ws.sh
reproin-dicom
reproin-dicom_job_20250618_113525_zodD_out
reproin-dicom_job_20250618_113525_zodD_work
```

## Delete a workspace

If you no longer need the data stored in a workspace, you should delete it.
To delete one or several workspaces, you can do so with the following command:

```console
$ ./release-ws.sh
```

You will then be asked to select the workspaces you want to delete from a list.
Use the up/down arrow keys and press `Tab` to select one or more workspaces.

```
Selected workspaces:
reproin-dicom_job_20250625_120537_yhW3_work
reproin-dicom_job_20250625_120537_yhW3_out
Release these workspaces? [y/n]
```

Type `y` to confirm the deletion of the selected workspaces.

Alternatively, you can specify the name of the workspace as an argument:

```console
$ ./release-ws.sh reproin-dicom_job_20250625_120537_yhW3_work
```

## Convert DICOM to BIDS

In this section we assume that
you have
uploaded a DICOM
dataset to the `reproin-dicom` workspace on the HPC cluster
according to the
[earlier section](hpc-upload).
If your use case is different, feel free to skip this section.

The next step is
to convert the DICOM files to BIDS format using
[HeuDiConv](https://heudiconv.readthedocs.io/en/latest/).
To do this for the `reproin-dicom` dataset, run the following command:

```console
$ ./run-job.sh reproin-dicom ./job/reproin-heudiconv/job.slurm
```

Below is another example based on the `social-detection-7t` dataset.
In this case, it is necessary to supply a custom hook script
(`hook.sh`) using the `-p` option.
This script is executed right before the SLURM `sbatch` command.
The script identifies the number of subjects in the dataset
and creates a SLURM job array. Meaning, if the dataset contains
DICOM files for 60 subjects, 60 SLURM jobs will be created
and processed in parallel:
```console
$ ./run-job.sh social-detection-7t-dicom ./job/social-detection-7t-heudiconv
```

If you are working with a different dataset, you may need to make
a copy of
[`reproin-heudiconv/job.slurm`](https://github.com/CENsBonn/tools/blob/main/job/social-detection-7t-heudiconv/job.slurm)
and adjust it to fit your dataset.
For example, you may need to change to a different
[HeuDiConv](https://github.com/nipy/heudiconv) heuristic.
A heuristic is specific
recipe for converting DICOM files to BIDS format.
In the case of the `reproin-dicom` dataset, the built-in `reproin` heuristic is
used. Depending on your dataset,
you may need to change this argument to refer to a different built-in
heuristic. If no built-in heuristic is suitable for your dataset, you need to
write your own custom heuristic in the form of a Python script.

A set of built-in heuristics can be found in the
[HeuDiConv repository](https://github.com/nipy/heudiconv/tree/2a15cdfa913d2288f08198db2196047c90711e1b/heudiconv/heuristics).
If you need to write a custom heuristic, upload it to the
[tools](https://github.com/CENsBonn/tools/tree/main/heuristics)
repository and pull the changes on the server side by running
`ssh marvin git -C ./tools pull`.
Then specify the path to the heuristic Python file in the SLURM script. Example:

```diff
- heuristic="reproin"
+ heuristic="$HOME/tools/job/social-detection-7t-heudiconv/heuristic.py"
```

Once you have run the HeuDiConv job, the output workspace will
contain the converted BIDS dataset.
The output workspace will have a name like `reproin_job_20250625_150428_ilac_out`.
You may want to rename this workspace to something simpler,
e.g. `reproin-bids`. See: {ref}`id-rename-a-workspace`.

(id-rename-a-workspace)=
## Rename a workspace

Let us assume you want to rename a workspace named `reproin_job_20250625_150428_ilac_out`
to `reproin-bids`. The renamed workspace should have an expiry time of `7` days.
Run the following command:

```console
$ ./rename-ws.sh reproin_job_20250625_150428_ilac_out reproin-bids 7
```

## Validate BIDS dataset

Validate your BIDS dataset using Bids Validator like so:

```console
$ ./bids-validate.sh reproin-bids
```

The first argument (`reproin-bids`) is the name of the workspace containing the
BIDS dataset.

## Run an fMRIPrep job

To run `fMRIPrep` on the uploaded dataset, you can use the following command:

```console
$ ./run-job.sh reproin fmriprep.slurm sub-001
```

The extra argument `sub-001` above is the label of the participant you would
like to process.

If you'd like to receive an email notification when the job finishes, you
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

### Job exits in `FAILED` status

Under certain conditions, the `apptainer` command of your SLURM job may fail
with exit code `1`. Below is an example of this:

```console
$ ./list-jobs.sh

[...]
 job_20250623_164344_Nhyr 2025-06-23T16:43:50   07:33:03     FAILED
```

Yet, the output of fMRIPrep does not contain any errors, and even reports
success:

```console
$ ./tail-output.sh jobs/job_20250623_164344_Nhyr/slurm.out

[...]
250624-00:16:51,945 nipype.workflow IMPORTANT:
	 fMRIPrep finished successfully!
[...]
```

The reason behind this `FAILED` status is currently not fully understood. The
behavior has been observed on at least fMRIPrep version `25.1.1`.
[One thread](https://neurostars.org/t/exit-code-1-despite-no-error-in-logs/30164/19?page=2)
suggests that it is due to a version warning, but this has been observed even
without any version warning in the output. It appears that the job successfully
exits with `COMPLETED` status if the `--fs-no-reconall` option is passed. This
would indicate that the failure arises from the FreeSurfer surface
preprocessing step.
