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

# fMRI analysis

## Notes

To include:

* MVPA: Multivariate pattern analysis: analyze patterns of brain activity across multiple voxels simultaneously rather tan one voxel at a time

That's the strength of Nilearn. Tools like SPM or FSL focus on univariate methods.

The following sections
are based on the
[Intro to GLM Analysis](https://nilearn.github.io/stable/auto_examples/00_tutorials/plot_single_subject_single_run.html)
tutorial by Nilearn.

## Setup

{doc}`Set up the Conda environment<local-setup-environment>`,
activate it, and install `nilearn`:

```console
$ conda activate cens
$ conda install -y -c conda-forge nilearn
```

## Research question

What part of the brain is activated when a person is listening to the sound of spoken words?

## Dataset


To address the research question,
we will start by loading the SPM auditory dataset.
This is a very simple dataset, making it suitable for beginners.

Run:

```python
from nilearn.datasets import fetch_spm_auditory
subject_data = fetch_spm_auditory()
```

This will trigger the dataset to be downloaded to the following directory:

```console
$ tree ~/nilearn_data/spm_auditory
/home/sebelino/nilearn_data/spm_auditory
└── MoAEpilot
    ├── CHANGES
    ├── dataset_description.json
    ├── README
    ├── sub-01
    │   ├── anat
    │   │   └── sub-01_T1w.nii
    │   └── func
    │       ├── sub-01_task-auditory_bold.nii
    │       └── sub-01_task-auditory_events.tsv
    └── task-auditory_bold.json
```

As we can see, this dataset is organized according to
[BIDS](https://bids.neuroimaging.io/).
The most important files are described below.

### `task-auditory_bold.json`
Sidecar JSON file for the functional BOLD scan.
Contains data about how the MRI scan was performed.

```console
$ cat ~/nilearn_data/spm_auditory/MoAEpilot/task-auditory_bold.json
```
```json
{
  "RepetitionTime": 7,
  "NumberOfVolumesDiscardedByScanner": 0,
  "NumberOfVolumesDiscardedByUser": 12,
  "TaskName": "Auditory",
  "TaskDescription": "The condition for successive blocks alternated between rest and auditory stimulation, starting with rest. Auditory stimulation was with bi-syllabic words presented binaurally at a rate of 60 per minute.",
  "Manufacturer": "Siemens",
  "ManufacturersModelName": "MAGNETOM Vision",
  "MagneticFieldStrength": 2
}
```

`RepetitionTime` is the time between successive MRI scans.
In essence, this means a full 3D snapshot of the entire brain was taken every 7 seconds.

`TaskDescription` describes what the participant was doing while he/she was
laying in the MRI scanner.
The text tells us that experiment was designed in the following way:

1. Participant is asked to lay still in the MRI scanner.
1. Scan begins.
1. Participant lays still in the scanner for $x$ seconds.
1. A sound clip is played.
1. Participant listens to the sound clip for $x$ seconds.
1. The sound clip is muted.
1. Steps 3-6 are repeated several times.

Given this experimental design, it is reasonable to expect that there will be
more brain activity in the parts of the brain responsible for auditory
processing whenever the participant is listening to the sound clip (step 5)
compared to when the participant is hearing nothing (step 3). Our goal is to
test this hypothesis.

(id-auditory-events-tsv)=
### `func/sub-01_task-auditory_events.tsv`
This file describes additional details about the experimental design.
Specifically, it describes the timing of experimental events.

```console
$ cat ~/nilearn_data/spm_auditory/MoAEpilot/sub-01/func/sub-01_task-auditory_events.tsv
```
```text
onset	duration	trial_type
42	42	listening
126	42	listening
210	42	listening
294	42	listening
378	42	listening
462	42	listening
546	42	listening
```

This file tells us that the participant was
subject to no auditory stimulus for the
first $x=42$ seconds,
then listened to a sound clip for the next $x=42$ seconds.
Then these two steps were repeated until a total of
$7\cdot(42+42) = 588$ seconds had passed.

### `anat/sub-01_T1w.nii`
High-resolution anatomical 3D image of the subject's brain.
You can plot 2D slices of the image by running:

```python
from nilearn.datasets import fetch_spm_auditory
subject_data = fetch_spm_auditory()
from nilearn.plotting import plot_anat, show

plot_anat(
    subject_data.anat,
    cbar_tick_format="%i",
)
show()
```

To get its data:

```python
from nilearn import image
from nilearn.datasets import fetch_spm_auditory

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.anat)
data = img.get_fdata()
voxel_dimensions = tuple(float(c) for c in img.header.get_zooms())
print(f"{data.shape=}")  # Prints (256,256,54)
print(f"{voxel_dimensions=}")  # Prints (1.0, 1.0, 3.0)
```

The output of the script above reveals that
the image is a 3D volume with consisting of
$256\times 256\times 54=3538944$
voxels in total, where each voxel is of size $1.0 \times 1.0 \times 3.0$ millimeter.
This means each voxel is three times longer in the inferior-superior direction
compared to the left-right and anterior-posterior directions.

To study a specific voxel, e.g.
$(128, 128, 28)$, you can run
`python plot_anat_voxel.py 128 128 28`
on the following script:

```python
import sys
from nilearn import image
from nilearn.plotting import plot_anat, show
import numpy as np
from nilearn.datasets import fetch_spm_auditory

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.anat)
data = img.get_fdata()

if len(sys.argv) != 4:
    print("Usage: python plot_anat_voxel.py <x> <y> <z>")
    sys.exit(1)

x, y, z = [int(t) for t in sys.argv[1:]]
voxel_mm = (img.affine @ np.array([x, y, z, 1]))[:3]
print(f"{data[(x, y, z)]=}")  # Prints 22.0
print(f"{voxel_mm=}")  # Prints (-1.0, -39.0, 12.0) mm
plot_anat(
  img,
  display_mode='ortho',
  cut_coords=voxel_mm[:3],
  title=f"Voxel {(x, y, z)}, intensity={data[(x, y, z)]}",
  annotate=True,
)
show()
```

The voxel $(128, 128, 28)$ is at the center of the 3D image. According to the
output above, this corresponds to the real-world scanner-space coordinates
$(-1.0, -39.0, 12.0)$ millimeters. In addition, the intensity of the voxel is
$22.0$. This is a relatively low value compared to surrounding voxels. A value
between 0 and 50 indicates this this voxel is inside the ventricles of the
brain, where there is cerebrospinal fluid instead of white/grey matter.
The plotted image confirms this.

```{figure} ./images/anat-mid-voxel-plot.png
:name: voxel-plot
```

### `func/sub-01_task-auditory_bold.nii`
The actual fMRI data.
Consists of a 4D image (3D space + time) of the subject's brain activity.
To study the data, run:

```python
from nilearn import image
from nilearn.datasets import fetch_spm_auditory

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.func)
data = img.get_fdata()
voxel_dimensions = tuple(float(c) for c in img.header.get_zooms())
print(f"{data.shape=}")  # Prints (64, 64, 64, 84)
print(f"{voxel_dimensions=}")  # Prints (3.0, 3.0, 3.0, 7.0)
```

The output shows that the data consists of
$64\times 64\times 64$ time series, one for each voxel.
Each time series in turn consists of 84 time points.
Furthermore, each voxel is $3.0 \times 3.0 \times 3.0$ millimeters,
and the time points are 7 seconds apart.

```{figure} ./images/voxel-sizes.png
:name: voxel-sizes

Blue cuboid: $1\times 1\times 3$ mm voxel of the anatomical image.
Red cube: $3\times 3\times 3$ mm voxel of the functional image.
```

Note that the anatomical image is $256\times 256\times 162\ \text{mm}^3$ in size,
whereas the functional image is $192\times 192\times 192\ \text{mm}^3$.
The functional block fits within the anatomical block in the $xy$ dimensions,
but covers a slightly larger range in the $z$ dimension.

```{figure} ./images/anat-vs-fmri-fov.png
:name: anat-vs-fmri-fov

Transparent cuboid: $256\times 256\times 162$ mm field-of-view of the anatomical image.
Blue cube: $192\times 192\times 192$ mm field-of-view of the functional image.
```

To output a GIF of a transverse section of the 4D image, run:

```python
import os
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, datasets

subject_data = datasets.fetch_spm_auditory()
img = image.load_img(subject_data.func)

z_slice = 32  # axial slice index
gif_filename = "fmri_timecourse.gif"
temp_dir = "/tmp/fmri_frames"
os.makedirs(temp_dir, exist_ok=True)

data = img.get_fdata()
n_volumes = data.shape[3]

frame_paths = []
for t in range(n_volumes):
    volume = data[:, :, z_slice, t]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.rot90(volume), cmap="gray", interpolation="nearest")
    ax.set_title(f"Time {t}")
    ax.axis("off")
    frame_path = os.path.join(temp_dir, f"frame_{t:03}.png")
    plt.savefig(frame_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    frame_paths.append(frame_path)

with imageio.get_writer(gif_filename, mode='I', duration=0.2) as writer:
    for frame_path in frame_paths:
        writer.append_data(imageio.imread(frame_path))

shutil.rmtree(temp_dir)
print(f"GIF saved to: {gif_filename}")
```

```{figure} ./images/fmri-animation.gif
:name: fmri-animation

GIF of a transverse section ($z=32$) of the 4D fMRI data.
```

From {numref}`fmri-animation` alone, it is difficult to tell whether the act of
listening to the sound clip affects the brain activity in this section of the
brain. If it does, then the difference in BOLD signal intensity is so low that
it is not visible to the naked eye. This is indeed often the case.

To study the time series of a specific voxel, e.g. $(32, 32, 32)$, run
`python plot_func_voxel.py 32 32 32` on the following script:

```python
import sys
from nilearn import image
from nilearn.plotting import plot_anat, plot_img, show
from nilearn.image import mean_img
import numpy as np
from nilearn.datasets import fetch_spm_auditory
import matplotlib.pyplot as plt

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.func)
data = img.get_fdata()

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <x> <y> <z>")
    sys.exit(1)

x, y, z = [int(e) for e in sys.argv[1:]]

if not (0 <= x < data.shape[0] and 0 <= y < data.shape[1] and 0 <= z < data.shape[2]):
    print(f"Voxel ({x}, {y}, {z}) is out of bounds. Data shape: {data.shape[:3]}")
    sys.exit(1)

time_series = data[x, y, z, :]
n_timepoints = time_series.shape[0]

print(time_series)

fig, ax = plt.subplots()
ax.plot(time_series, 'o-', markersize=2)
ax.set_ylim(bottom=0)
ax.set_title(f"Time series at voxel ({x}, {y}, {z})")
ax.set_xlabel("Time points")
ax.set_ylabel("Signal intensity")

block_size = 7
for start in range(0, n_timepoints, block_size * 2):
    listening_start = start + block_size
    listening_end = min(listening_start + block_size, n_timepoints)
    if listening_start < n_timepoints:
        ax.axvspan(listening_start, listening_end, color='lightblue', alpha=0.3, label='Listening' if start == 0 else "")

ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("voxel-timeseries.png", dpi=300)
plt.show()
```

```{figure} ./images/voxel-timeseries.png
:name: voxel-timeseries
:width: 70%

The time series of the voxel at $(32, 32, 32)$.
The shaded areas indicate the time periods when the participant was listening
to the sound clip.
```

The time series in {numref}`voxel-timeseries` corresponds to the pixel in the
very center of {numref}`fmri-animation`. As we can see, the BOLD signal
intensity appears to be relatively flat throughout the entire time series.
There is no obvious difference in intensity between the time periods when the
participant was listening to the sound clip and compared to when the
participant was at rest.

To proceed, we could systematically do a separate time series analysis for each
of the $64\cdot 64\cdot 64=262144$ voxels
to see if there is a significant difference in brain activity for certain voxels.
That would be a {ref}`univariate analysis<id-univariate-analysis>`.

(id-univariate-analysis)=
## Univariate analysis

Consider the time series for voxel $v=(32,32,32)$ in {numref}`voxel-timeseries`.
We need a statistical method to determine whether the intensity of the BOLD
signal is significantly different within the shaded area of the plot.

### Dummy variable regression

We will start by considering a simple linear dummy variable regression model.

Let $Y^v(t)$ be a random variable representing the BOLD signal intensity at voxel $v$
and time point $t$.
For example, we can see from {numref}`voxel-timeseries` that $y^v(0)=1259.0$ is
a realization of this random variable.

Let $x_{\text{listen}}(t) = 1$ if the participant was listening at time point $t$.
Let $x_{\text{listen}}(t) = 0$ if the participant was at rest at $t$.
From the {ref}`experimental design<id-auditory-events-tsv>`
and {numref}`voxel-timeseries`, we know that:

$$
\begin{align*}
    x_{\text{listen}}(0) = 0 \\
    \ldots \\
     x_{\text{listen}}(6) = 0 \\
    x_{\text{listen}}(7) = 1 \\
    \ldots \\
    x_{\text{listen}}(13) = 1 \\
    x_{\text{listen}}(14) = 0 \\
    \ldots \\
    x_{\text{listen}}(83) = 1
\end{align*}
$$

Let $\beta_{\text{listen}}^v$ be the regression coefficient for the
$x_{\text{listen}}^v$ dummy variable. We do not know $\beta_{\text{listen}}^v$, but we can estimate it
from the data.

Let $\beta_0^v$ be the regression intercept.
We do not know it, but we can estimate it from the data.

Finally, let $\varepsilon^v \sim \mathcal{N}(0,\sigma^2)$ be a random variable capturing random noise in the model.
We do not know the fixed variance $\sigma^2$, but we can estimate it from the data.

We can now make the following assumption: there is a linear relationship
between the signal intensity and the listening variable. The relationship
can be modeled as follows:

$$ Y^v(t) = \beta_0^v + \beta_{\text{listen}}^v x_{\text{listen}}(t) + \varepsilon^v $$

Note that the signal does not depend on $t$ aside from the dummy variable. We
can therefore express the relationship simply as:

$$ Y^v = \beta_0^v + \beta_{\text{listen}}^v x_{\text{listen}} + \varepsilon^v $$

We now make the following null hypothesis:
there is no significant difference in BOLD signal intensity
between the time periods when the participant was listening to the sound clip
compared to when the participant was at rest.
The alternative hypothesis is that there is a significant difference. These
hypotheses can be formally expressed as follows:

$$
\begin{align*}
    H_0: \beta_{\text{listen}} = 0 \\
    H_a: \beta_{\text{listen}} \neq 0
\end{align*}
$$

In order to attempt to reject $H_0$,
we must first fit the model to the time series data in {numref}`voxel-timeseries`.

To print the time series data shown in {numref}`voxel-timeseries`, run
`python print_func_voxel.py 32 32 32` on the following script:

```python
import sys
from nilearn import image
from nilearn.plotting import plot_anat, plot_img, show
from nilearn.image import mean_img
import numpy as np
from nilearn.datasets import fetch_spm_auditory
import matplotlib.pyplot as plt

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.func)
data = img.get_fdata()

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <x> <y> <z>")
    sys.exit(1)

x, y, z = [int(e) for e in sys.argv[1:]]

if not (0 <= x < data.shape[0] and 0 <= y < data.shape[1] and 0 <= z < data.shape[2]):
    print(f"Voxel ({x}, {y}, {z}) is out of bounds. Data shape: {data.shape[:3]}")
    sys.exit(1)

segment_size = 7
time_series = data[x, y, z, :]
time_series = list(((i // segment_size) % 2, float(y)) for i, y in enumerate(time_series))

print(time_series)
```

The output consists of a total of 84 data points of the form $(x_{\text{listen}}^v,y^v)$:

```python
[(0, 1259.0), (0, 1230.0), (0, 1241.0), (0, 1319.0), (0, 1296.0), (0, 1286.0), (0, 1274.0),
 (1, 1305.0), (1, 1273.0), (1, 1257.0), (1, 1257.0), (1, 1247.0), (1, 1280.0), (1, 1287.0),
 (0, 1247.0), (0, 1249.0), (0, 1236.0), (0, 1252.0), (0, 1255.0), (0, 1267.0), (0, 1254.0),
 (1, 1223.0), (1, 1297.0), (1, 1313.0), (1, 1252.0), (1, 1275.0), (1, 1270.0), (1, 1248.0),
 (0, 1283.0), (0, 1283.0), (0, 1271.0), (0, 1271.0), (0, 1272.0), (0, 1257.0), (0, 1274.0),
 (1, 1290.0), (1, 1276.0), (1, 1240.0), (1, 1259.0), (1, 1296.0), (1, 1237.0), (1, 1251.0),
 (0, 1291.0), (0, 1237.0), (0, 1247.0), (0, 1244.0), (0, 1231.0), (0, 1262.0), (0, 1227.0),
 (1, 1319.0), (1, 1318.0), (1, 1310.0), (1, 1284.0), (1, 1332.0), (1, 1248.0), (1, 1290.0),
 (0, 1238.0), (0, 1259.0), (0, 1293.0), (0, 1253.0), (0, 1296.0), (0, 1302.0), (0, 1279.0),
 (1, 1264.0), (1, 1320.0), (1, 1283.0), (1, 1294.0), (1, 1262.0), (1, 1252.0), (1, 1325.0),
 (0, 1299.0), (0, 1311.0), (0, 1237.0), (0, 1272.0), (0, 1224.0), (0, 1297.0), (0, 1255.0),
 (1, 1272.0), (1, 1243.0), (1, 1267.0), (1, 1286.0), (1, 1260.0), (1, 1289.0), (1, 1265.0)]
```

By plugging in the values above into our statistical model, we end up
with 84 equations. The first equation would be:

$$ 1259.0 = \beta_0^v + \varepsilon_0^v $$

In the eighth equation, we would set $x_{\text{listen}}=1$:

$$ 1305.0 = \beta_0^v + \beta_{\text{listen}}^v + \varepsilon_7^v $$

By minimizing the squared sum of residuals, we obtain estimates for the two coefficients:

$$ \hat\beta_0,\hat\beta_{\text{listen}} = \operatorname*{argmin}_{\beta_0,\beta_{\text{listen}}} (y_i^v - \beta_0 - \beta_{\text{listen}}\cdot x_{\text{listen},i})^2 $$

$$
\begin{align*}
    \hat\beta_{\text{listen}} &\approx 11.57 \\
    \hat\beta_0 &= 1265.0
\end{align*}
$$

We can additionally calculate the standard error and perform a t-test to
determine whether the coefficient is significantly different from zero.

```python
import sys
from nilearn import image
from nilearn.plotting import plot_anat, plot_img, show
from nilearn.image import mean_img
import numpy as np
from nilearn.datasets import fetch_spm_auditory
import matplotlib.pyplot as plt
from scipy import stats

subject_data = fetch_spm_auditory()
img = image.load_img(subject_data.func)
data = img.get_fdata()

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <x> <y> <z>")
    sys.exit(1)

x, y, z = [int(e) for e in sys.argv[1:]]

segment_size = 7
time_series = data[x, y, z, :]
time_series = list(((i // segment_size) % 2, float(y)) for i, y in enumerate(time_series))
print(time_series)

x = np.array([x for x, _ in time_series])
y = np.array([y for _, y in time_series])
n = len(x)

x_bar = np.mean(x)
y_bar = np.mean(y)

Sxx = np.sum((x - x_bar) ** 2)
Sxy = np.sum((x - x_bar) * (y - y_bar))

beta_1 = Sxy / Sxx
beta_0 = y_bar - beta_1 * x_bar

y_hat = beta_0 + beta_1 * x
residuals = y - y_hat

sigma_squared = np.sum(residuals**2) / (n - 2)
sigma = np.sqrt(sigma_squared)

se_beta_1 = sigma / np.sqrt(Sxx)
t_stat = beta_1 / se_beta_1
p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

print(f"beta_0 estimate:       {beta_0:.4f}")
print(f"beta_1 estimate:       {beta_1:.4f}")
print(f"standard error:        {se_beta_1:.4f}")
print(f"t-statistic:           {t_stat:.4f}")
print(f"degrees of freedom:    {n - 2}")
print(f"p-value:               {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: beta_1 ≠ 0 (significant)")
else:
    print("Fail to reject H0: insufficient evidence")
```

Output:

```
beta_0 estimate:       1265.0000
beta_1 estimate:       11.5714
standard error:        5.6110
t-statistic:           2.0623
degrees of freedom:    82
p-value:               0.042348
Reject H0: beta_1 ≠ 0 (significant)
```

The p-value is slightly below $0.05$, incidating that the
difference is barely significant, albeit under the weakest significance level.


