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
shown in {numref}`voxel-timeseries`.

```{figure} ./images/voxel-timeseries.png
:name: voxel-timeseries
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

Note that the signal does not depend on $t$, assuming we know the value of the
dummy variable $x_{\text{listen}}$. We
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

The p-value is slightly below $0.05$, incidating that the
difference is barely significant, albeit under the weakest significance level.


