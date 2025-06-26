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

# Setup

The following is a guide on how to connect to the HPC
cluster at CENs.
Before proceeding, make sure you have
{doc}`set up your local environment<local-setup-environment>`.

## Apply for access

To access the Marvin HPC cluster, you need to [apply for access](https://www.hpc.uni-bonn.de/en/systems/marvin).
Once you have access,
you should follow the instructions in the
[HPC wiki](https://wiki.hpc.uni-bonn.de/gaining_access)
on how to log in to the HPC cluster.
In a nutshell,
you should
generate a SSH keypair,
upload your public key to [FreeIPA](https://freeipa.hpc.uni-bonn.de/)
and connect to the cluster via SSH.

## Log in

Let us assume that you generated your keypair like so:

```console
$ ssh-keygen -t ed25519 -N "" -f ~/.ssh/marvin
```

Make sure that you are able to log in to the HPC cluster. Replace
`sebelin2_hpc` with your own HPC username in the command below:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de

        Welcome to the marvin-cluster!
        For more information check our wiki: https://wiki.hpc.uni-bonn.de
```

Then exit:

```console
[sebelin2_hpc@login02 ~]$ exit
```

Enter the
{ref}`tools<id-clone-tools-repository>`
repository and add an entry to your local `~/.ssh/config` by running the
command below:

```console
$ cd tools/
$ ./configure-remote-connection.sh sebelin2_hpc ~/.ssh/marvin
```

The command above simplifies your SSH setup.
Instead of having to type:

```console
$ ssh -i ~/.ssh/marvin sebelin2_hpc@marvin.hpc.uni-bonn.de
```

you now only have to type:

```console
$ ssh marvin
```

## Set up fMRIPrep

See {ref}`id-obtain-a-freesurfer-license` for how to obtain a FreeSurfer license.
Upload the license to the HPC cluster:

```console
$ ./upload-license.sh ./path/to/license.txt
```

Build the fMRIPrep Singularity image on the HPC cluster:

```console
$ ./build-fmriprep-image.sh
```

After running the two commands above, a directory containing two new files should
have appeared in your home directory:

```console
$ ssh marvin ls fmriprep/
fmriprep-25.1.1.simg
license.txt
```

These files will be necessary in order to run
fMRIPrep on the HPC cluster later.
