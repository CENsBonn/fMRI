# fmri

## Local development

Open a terminal and clone the repository:
```bash
git clone https://github.com/CENsBonn/fMRI
```
Alternatively:
```bash
git clone git@github.com:CENsBonn/fMRI
```

Install [Conda](https://conda-forge.org/download/)
if you haven't already.
Create an environment, activate it, and install the required packages:

```bash
conda create -n jbook
conda activate jbook
conda install -y -c conda-forge python
cd fMRI/
pip install -r requirements.txt 
```

To run the Jupyter Book locally, run:

```bash
make serve
```

A window should open in your browser at `http://localhost:8008`.

Now, try editing a file, e.g. `intro.md`. Upon saving the file, the Jupyter
Book should automatically rebuild and refresh in your browser.
