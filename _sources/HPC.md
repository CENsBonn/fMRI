# Instructions for using high-performance computer at Uni Bonn for fMRI analyses

## High-performance computer (HPC) at HRZ Uni Bonn: bonna
Login: jschultz
AG: ag-cens-jschultz
Password: censbonnaPASSWORD
Jobs running: https://bonna.hpc.uni-bonn.de/pun/sys/dashboard username is jschultz; in menu, click on Jobs / Active Jobs (open in Firefox)
Description: https://www.hpc.uni-bonn.de/en/hpc-infrastructure/bonna
Bonna wiki: https://bonna-wiki.hpc.uni-bonn.de
Sharing of resources: https://www.hpc.uni-bonn.de/en/topics/bonna_resource_sharing 
Nutzungsordnung: https://www.hpc.uni-bonn.de/en/hpc-infrastructure/vno_21_12_21.pdf 
HPC analytics lab (incl. Moritz Wolter and Sven Mallbach): https://www.dice.uni-bonn.de/hpca/en 

Log in using Terminal / Windows command line:
```
ssh jschultz@bonna.hpc.uni-bonn.de
```
The first time one does this on a given computer, it prints:

`
The authenticity of host 'bonna.hpc.uni-bonn.de (131.220.224.225)' can't be established.
ED25519 key fingerprint is SHA256:XvZ+e+fXl0FwQqWWhK1Q3jy3Pe4ZlvqI2wmjTtXgO9c.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? `
Then type:  
```
yes
```

It then prints:
`Warning: Permanently added 'bonna.hpc.uni-bonn.de' (ED25519) to the list of known hosts.  
(jschultz@bonna.hpc.uni-bonn.de) Password: 
`
[typed password in, then it printed:]

   Welcome to Bonna, the HPC cluster of the University of Bonn.

Then, we prepared the singularity container for fMRIprep, using the command:

```
VERSION="20.1.1"; singularity build fmriprep-"$VERSION".simg docker://poldracklab/fmriprep:"$VERSION"
```

Now we have an fMRIprep singularity container ready to run analyses.

Upload data from own computer using SFTP, using CyberDuck free software
Created shell scripts to run fMRIprep and a shell script for the slurm batch scheduling system (sbatch)


Example: replicate results of “Shared neural codes for visual and semantic information about familiar faces in a common representational space”, by [Castello Haxby Gobbini, PNAS 2021](https://www.pnas.org/doi/suppl/10.1073/pnas.2110474118).

Materials:
Data: https://openneuro.org/datasets/ds003834
Data to try hyperalignment: movie-watching of The Grand Budapest Hotel: https://openneuro.org/datasets/ds003017
Code: https://github.com/mvdoc/identity-decoding
Scripts for preproc., hyper alignment, GLM, MVPA, stats, visualisation (see https://github.com/mvdoc/identity-decoding/blob/main/README.md for details): https://github.com/mvdoc/identity-decoding/tree/main/scripts 
Previous code, with preprocessing based on fmri_ants_openfmri.py: https://www.github.com/mvdoc/famface


Then help Omar apply it to his analyses?

Here’s some code to run surface-based searchlight decoding on the Haxby 2001 dataset - maybe better to play with:
(Need to have python, pandas, sklearn, nilearn, maybe nipype?, download Haxby dataset)
https://nilearn.github.io/dev/auto_examples/02_decoding/plot_haxby_searchlight_surface.html

Code
fMRIprep installation documentation
Installation — fmriprep version documentation
All-in-one container system: Singularity
Executing with Singularity
Intro to Singularity
Introduction to Singularity — Singularity User Guide 3.8 documentation
Singularity code for Castello et al PNAS 2022
identity-decoding/singularity at main · mvdoc/identity-decoding
How to build a Singularity container
Build a Container — Singularity container 3.0 documentation
PyMVPA
Example Analyses and Scripts — PyMVPA 2.6.5.dev1 documentation
lib_prep_afni_surf.py: provides functionality to: - convert FreeSurfer surfaces to AFNI/SUMA format 
🔎 lib_prep_afni_surf - Google Search

