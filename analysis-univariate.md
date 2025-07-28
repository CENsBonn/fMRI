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

# Univariate analysis

In the previous section, we studied the time series of the voxel $v=(32,32,32)$,
shown in {numref}`voxel-timeseries2`.

```{figure} ./images/voxel-timeseries.png
:name: voxel-timeseries2
:width: 70%

The time series of the voxel at $(32, 32, 32)$.
The shaded areas indicate the time periods when the participant was listening
to the sound clip.
```

We need a statistical method to determine whether the intensity of the BOLD
signal is significantly different within the shaded area of the plot. To do
this, we must 1) design a statistical model of the BOLD signal and 2) test the
null hypothesis that the average signal intensity does not significantly change
during the listening phase under the assumptions of the model.

## Dummy variable regression

We will start by considering a simple linear dummy variable regression model.

Let $Y^v(t)$ be a random variable representing the BOLD signal intensity at voxel $v$
and time point $t$.
For example, we can see from {numref}`voxel-timeseries2` that $y^v(0)=1259.0$ is
a realization of this random variable.

Let $x_{\text{listen}}(t) = 1$ if the participant was listening at time point $t$.
Let $x_{\text{listen}}(t) = 0$ if the participant was at rest at $t$.
From the {ref}`experimental design<id-auditory-events-tsv>`
and {numref}`voxel-timeseries2`, we know that:

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

Finally, let $\varepsilon^v(t) \sim \mathcal{N}(0,\sigma^2)$ be a random variable capturing random noise in the model at time point $t$.
We do not know the fixed variance $\sigma^2$, but we can estimate it from the data.

We can now make the following assumption: there is a linear relationship
between the signal intensity and the listening variable. The relationship
can be modeled as follows:

$$ Y^v(t) = \beta_0^v + \beta_{\text{listen}}^v x_{\text{listen}}(t) + \varepsilon^v(t) $$

Note that the signal does not depend on $t$, assuming we know the value of the
dummy variable $x_{\text{listen}}$ and the noise term. We
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
    H_0: \beta_{\text{listen}}^v = 0 \\
    H_a: \beta_{\text{listen}}^v \neq 0
\end{align*}
$$

In order to attempt to reject $H_0$,
we must first fit the model to the time series data in {numref}`voxel-timeseries2`.

To print the time series data shown in {numref}`voxel-timeseries2`, run
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
with 84 equations. The first two equations would be:

$$
\begin{align*}
1259.0 &= \beta_0^v + \varepsilon_0^v \\
1230.0 &= \beta_0^v + \varepsilon_1^v
\end{align*}
$$

In the eighth equation, we would set $x_{\text{listen}}=1$, resulting in:

$$ 1305.0 = \beta_0^v + \beta_{\text{listen}}^v + \varepsilon_7^v $$

We can combine all 84 equations into matrix form:

$$
\begin{bmatrix}1259.0 \\ 1230.0 \\ \vdots \\ 1265.0\end{bmatrix} =
\begin{bmatrix}1 & 0 \\ 1 & 0 \\ \vdots & \vdots \\ 1 & 1\end{bmatrix}
\begin{bmatrix}\beta_0^v \\ \beta_{\text{listen}}^v\end{bmatrix} +
\begin{bmatrix}\varepsilon_0^v \\ \varepsilon_1^v \\ \vdots \\ \varepsilon_{83}^v\end{bmatrix}
$$

which we will denote more compactly:

$$ \mathbf{y}^v = X\boldsymbol{\beta}^v + \boldsymbol{\varepsilon}^v $$

Here, $X$ denotes the design matrix.

We want to find a pair of estimates for the two coefficients which minimizes
the squared sum of residuals, i.e.:

$$ (\hat\beta_0^v,\hat\beta_{\text{listen}}^v) = \operatorname*{argmin}_{\beta_0,\beta_{\text{listen}}} \sum_{i=0}^{83} (y_i^v - \beta_0 - \beta_{\text{listen}}\cdot x_{\text{listen},i})^2 $$

which can be written more compactly as:

$$ \hat{\boldsymbol\beta}^v = \operatorname*{argmin}_{\boldsymbol\beta} ||\mathbf{y}^v - X\boldsymbol{\beta}||^2 $$

By differentiating with respect to $\boldsymbol\beta$ and setting the derivative to zero,
we obtain the ordinary-least squares solution:

$$ \hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top \mathbf{y} $$

Plugging in the values, we get:

$$
\begin{align*}
    \hat\beta_{\text{listen}} &\approx 11.57 \\
    \hat\beta_0 &= 1265.0
\end{align*}
$$

We can additionally calculate the standard error and perform a t-test to
determine whether the coefficient is significantly different from zero.
Run `python test_func_voxel.py 32 32 32` on the following script:

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

print(f"beta_0 estimate:         {beta_0:.4f}")
print(f"beta_listening estimate: {beta_1:.4f}")
print(f"standard error:          {se_beta_1:.4f}")
print(f"t-statistic:             {t_stat:.4f}")
print(f"degrees of freedom:      {n - 2}")
print(f"p-value:                 {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: beta_listening ≠ 0 (significant)")
else:
    print("Fail to reject H0: insufficient evidence")
```

Output:

```
beta_0 estimate:         1265.0000
beta_listening estimate: 11.5714
standard error:          5.6110
t-statistic:             2.0623
degrees of freedom:      82
p-value:                 0.042348
Reject H0: beta_listening ≠ 0 (significant)
```

The p-value is slightly below $0.05$, indicating that we reject the null
hypothesis at the 5 % level.
Does this prove that this voxel is activated during the listening phase?
Or is the low p-value a result of model misspecification?

## General Linear Model

The dummy variable regression model is relatively simple, but makes a number of
assumptions which may not accurately reflect reality:

* The errors $\varepsilon_0, \ldots, \varepsilon_{83}$ are independent and
homoscedastic (i.e. they have equal variance).
However, in fMRI analysis, it is common to assume that they are
[autocorrelated](https://arxiv.org/pdf/0906.3662).
* The BOLD signal intensity changes exactly at the same time as the stimulus
toggles on or off. In reality, the BOLD signal is delayed and dispersed over
time in accordance with the [hemodynamic response function (HRF)](https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/Statistics/03_Stats_HRF_Overview.html).
* The model does not account for fMRI [drift](https://pubmed.ncbi.nlm.nih.gov/10329292/).
* Each voxel is analyzed separately. The time series data for voxel A is not used
to estimate the model coefficients for voxel B.
This can be addressed by multivariate analysis.

We can remedy some of these limitations by using a more general statistical
model based on AR(1) noise autocorrelation, the hemodynamic response function
(HRF), and drift. This can be easily achieved using `FirstLevelModel` from the
Nilearn library.
Run `python test_flm_voxel.py 32 32 32` on the following script:

```python
import sys
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_spm_auditory
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.contrasts import compute_contrast
from nilearn.image import load_img
from scipy.stats import norm

subject_data = fetch_spm_auditory()
img = load_img(subject_data.func)
n_scans = img.shape[-1]

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <x> <y> <z>")
    sys.exit(1)

x, y, z = [int(e) for e in sys.argv[1:]]

events = pd.read_table(subject_data.events)

glm = FirstLevelModel(
    t_r=7,
    noise_model="ar1",
    standardize=False,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=0.01,
)
glm.fit(img, events)

eff_map = glm.compute_contrast("listening", output_type="effect_size")
z_map = glm.compute_contrast("listening", output_type="z_score")

beta_1 = eff_map.get_fdata()[x, y, z]
z_score = z_map.get_fdata()[x, y, z]

p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Beta (listening): {beta_1:.4f}")
print(f"Z-score:          {z_score:.4f}")
print(f"P-value:          {p_value:.6f}")
```

Output:

```
Beta (listening): -0.7337
Z-score:          -1.8257
P-value:          0.067897
```

This time, the p-value is $0.068$, indicating that we failed to reject the null
hypothesis.

In the dummy regression model, our design matrix $X$ consisted of only two
columns: one for the intercept and one for the dummy variable
$x_{\text{listen}}$. In the Nilearn GLM, the matrix has been extended with
additional columns for the purpose of modeling drift:

```python
>>> glm.design_matrices_[0]
       listening   drift_1   drift_2   drift_3   drift_4   drift_5   drift_6   drift_7   drift_8   drift_9  drift_10  drift_11  constant
0.0     0.000000  0.154276  0.154195  0.154061  0.153872  0.153629  0.153333  0.152983  0.152580  0.152123  0.151613  0.151050       1.0
7.0     0.000000  0.154061  0.153333  0.152123  0.150435  0.148273  0.145644  0.142558  0.139023  0.135050  0.130652  0.125844       1.0
14.0    0.000000  0.153629  0.151613  0.148273  0.143637  0.137746  0.130652  0.122417  0.113112  0.102820  0.091628  0.079637       1.0
21.0    0.000000  0.152983  0.149046  0.142558  0.133631  0.122417  0.109109  0.093934  0.077152  0.059049  0.039937  0.020141       1.0
28.0    0.000000  0.152123  0.145644  0.135050  0.120639  0.102820  0.082094  0.059049  0.034336  0.008652 -0.017276 -0.042717       1.0
...          ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
553.0   0.809244 -0.152123  0.145644 -0.135050  0.120639 -0.102820  0.082094 -0.059049  0.034336 -0.008652 -0.017276  0.042717       1.0
560.0   1.129980 -0.152983  0.149046 -0.142558  0.133631 -0.122417  0.109109 -0.093934  0.077152 -0.059049  0.039937 -0.020141       1.0
567.0   1.023340 -0.153629  0.151613 -0.148273  0.143637 -0.137746  0.130652 -0.122417  0.113112 -0.102820  0.091628 -0.079637       1.0
574.0   1.001027 -0.154061  0.153333 -0.152123  0.150435 -0.148273  0.145644 -0.142558  0.139023 -0.135050  0.130652 -0.125844       1.0
581.0   1.000000 -0.154276  0.154195 -0.154061  0.153872 -0.153629  0.153333 -0.152983  0.152580 -0.152123  0.151613 -0.151050       1.0

[84 rows x 13 columns]
```

In addition, the values in the `listening` column are not perfectly binary, but
have been convolved to account for the HRF.

Rather than testing each voxel one by one, let us test all voxels at once and
use `plot_stat_map` to plot the most statistically significant ones:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.datasets import fetch_spm_auditory
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.contrasts import compute_contrast
from nilearn.image import load_img
from nilearn.plotting import plot_stat_map, show
from nilearn.image import mean_img
from nilearn.glm import threshold_stats_img
from scipy.stats import norm

subject_data = fetch_spm_auditory()
img = load_img(subject_data.func)

events = pd.read_table(subject_data.events)

glm = FirstLevelModel(
    t_r=7,
    noise_model="ar1",
    standardize=False,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=0.01,
)
glm.fit(img, events)

z_map = glm.compute_contrast("listening", output_type="z_score")

fmri_img = subject_data.func
mean_image = mean_img(subject_data.func[0], copy_header=True)

plotting_config = {
    "bg_img": mean_image,
    "display_mode": "z",
    "cut_coords": 3,
    "black_bg": True,
    "cmap": "inferno",
}

fig = plt.figure(figsize=(10, 4))

clean_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.001,
    height_control="fpr",
    two_sided=False,
)

plot_stat_map(
    clean_map,
    threshold=threshold,
    title=f"listening > rest (Uncorrected p<0.001; threshold: {threshold:.3f}",
    figure=fig,
    **plotting_config,
)
show()
```

```{figure} ./images/zmap.png
:name: zmap
:width: 100%

The z-map of the `listening > rest` contrast. Only voxels yielding a p-value below 0.001 (z-score $\approx 3.29$) are shown.
```

Figure {numref}`zmap` suggests that there are at least two clusters of voxels
where the z-scores are especially high. It is possible that these match the
regions of the brain where auditory processing is done. The remaining voxels
might be false positives. It is possible to discard isolated highlighted voxels
using `cluster_threshold`, as has been done in {numref}`zmap-animation`.

```{figure} ./images/zmap-animation.gif
:name: zmap-animation
:width: 50%

Animation of the z-map for all transverse slices that contain at least one voxel above the z-score threshold.
```

Lastly, we can use
[AtlasReader](https://github.com/miykael/atlasreader)
to label the regions of the z-map:

```python
#!/usr/bin/env python

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_spm_auditory
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

import atlasreader

# Run GLM analysis
subject_data = fetch_spm_auditory()
events = pd.read_table(subject_data.events)

glm = FirstLevelModel(t_r=7, noise_model="ar1", standardize=False,
                     hrf_model="spm", drift_model="cosine", high_pass=0.01)
glm.fit(subject_data.func, events)

z_map = glm.compute_contrast("listening", output_type="z_score")
thresholded_map, threshold = threshold_stats_img(
    z_map, alpha=0.05, height_control="fdr", cluster_threshold=10, two_sided=False
)

# Save statistical map for AtlasReader
temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
thresholded_map.to_filename(temp_file.name)

try:
    # Create output directory
    output_dir = tempfile.mkdtemp()
    
    # Run AtlasReader
    result = atlasreader.create_output(
        filename=temp_file.name,
        cluster_extent=10,
        atlas='default',
        voxel_thresh=threshold,
        direction='pos',
        prob_thresh=5,
        outdir=output_dir
    )
    
    # Read results
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if csv_files:
        results_path = os.path.join(output_dir, csv_files[0])
        atlas_results = pd.read_csv(results_path)
        
        # Display results
        for idx, cluster in atlas_results.iterrows():
            print(f"Cluster {idx + 1}:")
            print(f"  Peak coordinates (MNI): ({cluster.get('peak_x', 'N/A')}, {cluster.get('peak_y', 'N/A')}, {cluster.get('peak_z', 'N/A')})")
            print(f"  Cluster size: {cluster.get('volume_mm', cluster.get('cluster_size_mm3', 'N/A'))} mm³")
            
            # Show atlas labels
            atlas_columns = [col for col in atlas_results.columns if col in ['aal', 'harvard_oxford', 'desikan_killiany']]
            
            if atlas_columns:
                print(f"  Atlas labels:")
                for col in atlas_columns:
                    label = cluster.get(col, 'N/A')
                    if pd.notna(label) and label != '' and label != 'N/A':
                        print(f"    {col}: {label}")
            print()

finally:
    # Cleanup
    try:
        os.unlink(temp_file.name)
        import shutil
        if 'output_dir' in locals():
            shutil.rmtree(output_dir)
    except:
        pass
```

Output:

```
Cluster 1:
  Peak coordinates (MNI): (-60.0, -6.0, 42.0)
  Cluster size: 3888.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 20.0% Left_Precentral_Gyrus; 11.0% Left_Postcentral_Gyrus

Cluster 2:
  Peak coordinates (MNI): (60.0, 0.0, 36.0)
  Cluster size: 1620.0 mm³
  Atlas labels:
    aal: Postcentral_R
    desikan_killiany: ctx-rh-precentral
    harvard_oxford: 77.0% Right_Precentral_Gyrus; 5.0% Right_Postcentral_Gyrus

Cluster 3:
  Peak coordinates (MNI): (36.0, -3.0, 15.0)
  Cluster size: 1161.0 mm³
  Atlas labels:
    aal: Insula_R
    desikan_killiany: ctx-rh-insula
    harvard_oxford: 57.0% Right_Insular_Cortex; 13.0% Right_Central_Opercular_Cortex

Cluster 4:
  Peak coordinates (MNI): (45.0, -18.0, 57.0)
  Cluster size: 972.0 mm³
  Atlas labels:
    aal: Precentral_R
    desikan_killiany: Unknown
    harvard_oxford: 39.0% Right_Postcentral_Gyrus; 16.0% Right_Precentral_Gyrus

Cluster 5:
  Peak coordinates (MNI): (-15.0, -60.0, 66.0)
  Cluster size: 864.0 mm³
  Atlas labels:
    aal: Precuneus_L
    desikan_killiany: Left-Cerebral-White-Matter
    harvard_oxford: 50.0% Left_Lateral_Occipital_Cortex_superior_division; 24.0% Left_Superior_Parietal_Lobule

Cluster 6:
  Peak coordinates (MNI): (66.0, 15.0, 27.0)
  Cluster size: 837.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 7:
  Peak coordinates (MNI): (51.0, 30.0, 27.0)
  Cluster size: 675.0 mm³
  Atlas labels:
    aal: Frontal_Inf_Tri_R
    desikan_killiany: Unknown
    harvard_oxford: 65.0% Right_Middle_Frontal_Gyrus; 6.0% Right_Inferior_Frontal_Gyrus_pars_triangularis

Cluster 8:
  Peak coordinates (MNI): (-12.0, -69.0, 51.0)
  Cluster size: 540.0 mm³
  Atlas labels:
    aal: Precuneus_L
    desikan_killiany: ctx-lh-superiorparietal
    harvard_oxford: 30.0% Left_Lateral_Occipital_Cortex_superior_division; 20.0% Left_Precuneous_Cortex

Cluster 9:
  Peak coordinates (MNI): (-12.0, -15.0, 93.0)
  Cluster size: 513.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 10:
  Peak coordinates (MNI): (27.0, -51.0, 75.0)
  Cluster size: 432.0 mm³
  Atlas labels:
    aal: Parietal_Sup_R
    desikan_killiany: Unknown
    harvard_oxford: 7.0% Right_Superior_Parietal_Lobule

Cluster 11:
  Peak coordinates (MNI): (60.0, 21.0, 75.0)
  Cluster size: 432.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 12:
  Peak coordinates (MNI): (-3.0, -27.0, 90.0)
  Cluster size: 378.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 13:
  Peak coordinates (MNI): (-24.0, -24.0, 90.0)
  Cluster size: 378.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 14:
  Peak coordinates (MNI): (60.0, -18.0, 27.0)
  Cluster size: 378.0 mm³
  Atlas labels:
    aal: SupraMarginal_R
    desikan_killiany: ctx-rh-supramarginal
    harvard_oxford: 46.0% Right_Postcentral_Gyrus; 24.0% Right_Supramarginal_Gyrus_anterior_division

Cluster 15:
  Peak coordinates (MNI): (45.0, -18.0, 78.0)
  Cluster size: 378.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 16:
  Peak coordinates (MNI): (63.0, -6.0, 27.0)
  Cluster size: 378.0 mm³
  Atlas labels:
    aal: Postcentral_R
    desikan_killiany: Right-Cerebral-White-Matter
    harvard_oxford: 51.0% Right_Postcentral_Gyrus; 22.0% Right_Precentral_Gyrus

Cluster 17:
  Peak coordinates (MNI): (-27.0, -3.0, 15.0)
  Cluster size: 351.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Left-Cerebral-White-Matter
    harvard_oxford: 30.0% Left_Putamen

Cluster 18:
  Peak coordinates (MNI): (21.0, 18.0, 93.0)
  Cluster size: 351.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 19:
  Peak coordinates (MNI): (21.0, -72.0, 51.0)
  Cluster size: 324.0 mm³
  Atlas labels:
    aal: Parietal_Sup_R
    desikan_killiany: Right-Cerebral-White-Matter
    harvard_oxford: 59.0% Right_Lateral_Occipital_Cortex_superior_division

Cluster 20:
  Peak coordinates (MNI): (18.0, -18.0, 96.0)
  Cluster size: 297.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label

Cluster 21:
  Peak coordinates (MNI): (9.0, -33.0, 90.0)
  Cluster size: 297.0 mm³
  Atlas labels:
    aal: no_label
    desikan_killiany: Unknown
    harvard_oxford: 0% no_label
```

From the output above, it is not obvious that the highlighted voxels are part
of the auditory cortex.
It could be that the `cluster_threshold` is too high
and the auditory regions of interest are too small, especially
considering the coarse resolution of the data (3 mm voxels).

