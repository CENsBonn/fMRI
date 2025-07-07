# fmri

## Local development

Open a terminal or WSL and clone the repository:
```bash
git clone https://github.com/CENsBonn/fMRI
```
Alternatively:
```bash
git clone git@github.com:CENsBonn/fMRI
```

Install [Conda](https://conda-forge.org/download/)
if you haven't already.
If you are using WSL, you can do:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda"
rm -f miniconda.sh
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
```
Then restart the terminal.
Create an environment, activate it, and install the required packages:

```bash
conda create -y -n jbook
conda activate jbook
conda install -y -c conda-forge python
cd fMRI/
pip install -r requirements.txt 
sudo apt install make
```

To run the Jupyter Book locally, run:

```bash
make serve

[I 250707 10:21:03 server:331] Serving on http://localhost:8008
```

Open http://localhost:8008 in your browser.

![image](https://github.com/user-attachments/assets/1973f3de-939a-4d63-a295-4a8820ae173b)

Now, try editing a file, e.g. `intro.md`.
Upon saving the file, the Jupyter
Book should automatically rebuild and refresh in your browser
after a couple of seconds.
