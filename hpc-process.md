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

(id-convert-dicom-to-bids)=
## Convert DICOM to BIDS

In this section we assume that
you have
uploaded a DICOM
dataset to the `reproin-dicom` workspace on the HPC cluster
according to the
{ref}`earlier section <id-upload-dicom-dataset>`.
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

To run `fMRIPrep` on the uploaded `reproin`, you can use the following command:

```console
$ ./run-job.sh reproin-bids ./job/reproin-fmriprep
```

If you are working with the `social-detection-7t` dataset, run:

```console
$ ./run-job.sh social-detection-7t-bids ./job/social-detection-7t-fmriprep
```

As the `social-detection-7t-bids` dataset contains 60 subjects, 60 jobs will be
spawned and processed in parallel. Each job takes 7-8 hours to complete.

If your dataset is different, make a copy of
[`job/social-detection-7t-fmriprep/`](https://github.com/CENsBonn/tools/tree/main/job/social-detection-7t-heudiconv)
and adjust the scripts to fit your dataset.

During the execution of a job, the `output/` and `work/` directories
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

## Email notifications

If you'd like to receive an email notification when a job finishes, you
can edit your `job.slurm` and add a `--mail-user` option before running it:

```diff
 #SBATCH --mem=32000
 #SBATCH --export=NONE
 #SBATCH --mail-type=END,FAIL,TIME_LIMIT
+#SBATCH --mail-user=sebelin2@uni-bonn.de
 #SBATCH --time=10:00:00
```

## Troubleshooting

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
