# fMRI

## Local development

Open a terminal or WSL and clone the repository:
```bash
git clone https://github.com/CENsBonn/fMRI
```
Alternatively, if you intend to contribute changes  to the repository, you need to
[generate an SSH keypair](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux#generating-a-new-ssh-key)
and run:
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
conda install -y -c conda-forge make
cd fMRI/
pip install -r requirements.txt 
```

To run the Jupyter Book locally, run:

```bash
conda activate jbook
make serve

[I 250707 10:21:03 server:331] Serving on http://localhost:8008
```

Open http://localhost:8008 in your browser.

![image](https://github.com/user-attachments/assets/1973f3de-939a-4d63-a295-4a8820ae173b)

Now, try editing a file, e.g. `intro.md`.
Upon saving the file, the Jupyter
Book should automatically rebuild and refresh in your browser
after a couple of seconds.

After editing the file, commit and push the changes:

```bash
git add .
git commit -m "docs: Update intro.md"
git push
```
