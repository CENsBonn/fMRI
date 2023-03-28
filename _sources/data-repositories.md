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

# Where can I get data?

Thanks to the formation of common data acquisition projects and the 
popularization of data sharing through repositories, more and more high-quality datasets can be accessed online.

```{Note}
Data to download are often in BIDS format, which is explained [here](https://bids.neuroimaging.io/).
```

## Datasets and repositories

### Websites linking to datasets and analyses

- A website with detailed tutorials about how to analyse movie data: [Naturalistic data](https://naturalistic-data.org).
- A platform for fast and flexible re-analysis of (naturalistic) fMRI studies, with 13 datasets and downloadable code: [Neuroscout](https://neuroscout.org/datasets); see also this [paper](https://doi.org/10.7554/eLife.79277).

### Multi-method datasets

- The [Human Connectome Project](https://www.humanconnectome.org/) contains a dataset of [1200 subjects aged 22-35](https://www.humanconnectome.org/study/hcp-young-adult).
- A large-scale biomedical database and research resource, containing in-depth genetic and health information from half a million UK participants: [UK Biobank](https://www.ukbiobank.ac.uk).
- fMRI data from 3 people looking at 8740 images of 720 objects, MEG data from 4 participants looking at 22448 images of 1854 objects, 4.7 mio similarity ratings from 12340 participants. The [THINGS-data](https://things-initiative.org) database. See the [paper](https://doi.org/10.7554/eLife.82580).
- Data from 12 2-hour sessions from 10 participants, with 4 T1, T2 and angiogram images, 8 MR venograms, 5 hours of resting-state, >5.5 hours of task fMRI data from 3 tasks, + neuropsychological tests: [Midnight Scan Club](https://openneuro.org/datasets/ds000224).
- One participant scanned repeatedly during a whole year: [MyConnectome](http://www.myconnectome.org).
- Ongoing initiative to create a biobank with [data from 10,000 New York City area children and adolescents (ages 5-21)](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/About.html).

### fMRI BOLD datasets

#### Responses to images

- 7T data from 8 participants presented with 1000s of images of natural scenes over 30-40 sessions, scanned at 1.8^3mm resolution: the [Natural Scenes Dataset](http://naturalscenesdataset.org). See also the [paper](https://doi.org/10.1038/s41593-021-00962-x).
- 4 participants observing 5,254 images over 15 sessions, slow event-related paradigm, 3T: [bold5OOO](https://bold5000-dataset.github.io/website/).
- Large assembly of datasets for computational neuroscience, such as [vim-1](https://crcns.org/data-sets/vc/vim-1) by the Collaborative Research in Computational Neuroscience project of the Redwood Center for Theoretical Neuroscience, UC Berkeley.

#### Responses to videos
- Subset of Human Connectome Project dataset: 184 participants (almost all twins/siblings) scanned at 7T (1.6mm3, TR 1s) for 4 runs of 15 minutes, each containing 3-4 video clips: [see the protocols](https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging).
- Data from 25 subjects who watched part of the movie [The Grand Budapest Hotel](https://doi.org/10.18112/openneuro.ds003017.v1.0.2); also on [DataLad](http://datasets.datalad.org/?dir=/labs/gobbini). Also avaibale are [Presentation, preprocessing, and analyses scripts](https://github.com/mvdoc/budapest-fmri-data). For details, see the [paper](https://www.nature.com/articles/s41597-020-00735-4).
- A single individual exposed to 30 episodes of BBC’s Doctor Who (TR=700 ms, 118,000 whole-brain volumes, approx. 23 h; the training set) + 500 volumes (5 min) of repeated short clips (test set, 22 repetitions), recorded with fixation over a period of six months: [DoctorWho](https://data.donders.ru.nl/collections/di/dcc/DSC_2018.00082_134?0).
- Participants watching the movie Forrest Gump, and more data: [StudyForrest](http://www.studyforrest.org).
- Responses to>1000 3s-videos with metadata (only a preprint so far, database not yet accessible): [Bold moments](https://www.biorxiv.org/content/10.1101/2023.03.12.530887v1.full.pdf).
- Participants watching the movies Sherlock and Merlin (18 participants each): [Sherlock_Merlin](https://openneuro.org/datasets/ds001110/versions/00003).
- 12 participants watching 40 naturalistic clips of behaving animals, by [Nastase et al.](https://openneuro.org/datasets/ds000233/versions/1.0.1), see [paper 1](https://doi.org/10.1093/cercor/bhx138) and [2](https://doi.org/10.3389/fnins.2018.00316).

#### Responses to spoken stories

- 345 subjects, 891 functional scans, and 27 diverse stories of varying duration totaling ~4.6 hours of unique stimuli (~43,000 words): the [Narratives fMRI dataset](https://www.nature.com/articles/s41597-021-01033-3), see this [paper](https://doi.org/10.1038/s41597-021-01033-3).

#### Responses during different tasks

- Data from 12 participants performing about 12 different tasks: the [Individual Brain Charting](https://www.nature.com/articles/sdata2018105) project.
- 94 participants performing a 5-minute "cognitive localizer" experiment with 8 tasks (including visual perception, finger tapping, language, and math): [The Brainomics/Localizer database](https://osf.io/vhtf6/files/osfstorage); [the same data preprocessed using fMRIprep](https://gin.g-node.org/ljchang/Localizer); [the paper describing the experiment](https://doi.org/10.1186/1471-2202-8-91); [the paper describing the dataset](https://doi.org/10.1016/j.neuroimage.2015.09.052).

## More general links
### Automated meta-analyses
- The most popular is [NeuroSynth](https://neurosynth.org/).

### General repositories
[Openneuro](https://openneuro.org/), [NeuroVault](https://neurovault.org/), [academictorrents](https://academictorrents.com/browse.php?search=fmri).


