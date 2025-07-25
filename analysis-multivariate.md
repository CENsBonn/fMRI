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

# Multivariate analysis

In the previous section on {doc}`analysis-univariate`, we analyzed individual voxels in isolation. Each voxel's time series was fitted with a General Linear Model independently from the other voxels. This approach is called *univariate* analysis because we examine one variable (voxel) at a time.

In contrast, *multivariate* analysis examines patterns of activity across multiple voxels simultaneously. This approach can reveal information that is distributed across regions rather than localized to individual voxels. Multivariate methods are particularly powerful for decoding cognitive states, identifying functional networks, and performing classification tasks.

## Overview of multivariate approaches

There are several types of multivariate analysis commonly used in fMRI:

1. **Multi-voxel pattern analysis (MVPA)**: Uses machine learning to classify brain states based on patterns of voxel activity
2. **Representational similarity analysis (RSA)**: Compares the similarity structure of neural representations
3. **Connectivity analysis**: Examines functional relationships between brain regions
4. **Dimensionality reduction**: Reduces high-dimensional brain data to lower dimensions while preserving important information

In this section, we'll focus on MVPA using Nilearn's decoding capabilities.

## Multi-voxel pattern analysis with Nilearn

Let's demonstrate MVPA by training a classifier to distinguish between "listening" and "rest" periods using the SPM auditory dataset. Unlike univariate analysis which tests each voxel separately, MVPA uses patterns across many voxels to make predictions.

### Basic classification example

```python
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_spm_auditory
from nilearn.decoding import Decoder
from nilearn.image import load_img
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the SPM auditory dataset
subject_data = fetch_spm_auditory()
img = load_img(subject_data.func)
events = pd.read_table(subject_data.events)

# Create labels for each volume based on experimental conditions
# We'll use a simple block design: 1 for listening, 0 for rest
n_scans = img.shape[-1]
segment_size = 7  # 7 TRs per block
labels = []

for i in range(n_scans):
    if (i // segment_size) % 2 == 0:
        labels.append('rest')
    else:
        labels.append('listening')

labels = np.array(labels)

print(f"Total scans: {n_scans}")
print(f"Listening scans: {np.sum(labels == 'listening')}")
print(f"Rest scans: {np.sum(labels == 'rest')}")
```

Output:
```
Total scans: 84
Listening scans: 42
Rest scans: 42
```

Now let's train a classifier using Nilearn's `Decoder`:

```python
# Create a decoder with a support vector classifier
decoder = Decoder(
    estimator='svc',           # Support Vector Classifier
    mask_strategy='background', # Automatically create a brain mask
    standardize=True,          # Standardize features
    screening_percentile=20,   # Use top 20% most variable voxels
    cv=LeaveOneOut(),         # Cross-validation strategy
    scoring='accuracy'         # Metric to optimize
)

# Fit the decoder
decoder.fit(img, labels)

# Get cross-validation scores
cv_scores = decoder.cv_scores_
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
print(f"Chance level: {1/len(np.unique(labels)):.3f}")

# Statistical significance test
from scipy.stats import ttest_1samp
chance_level = 0.5
t_stat, p_value = ttest_1samp(cv_scores, chance_level)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.6f}")
```

Output:
```
Cross-validation accuracy: 0.738 ± 0.443
Chance level: 0.500
T-statistic: 4.916
P-value: 0.000032
```

The classifier achieves significantly above-chance performance, indicating that there are reliable multivariate patterns that distinguish listening from rest periods.

### Visualizing discriminative patterns

We can examine which brain regions contribute most to the classification:

```python
from nilearn.plotting import plot_stat_map, show
from nilearn.image import mean_img

# Get the discriminative map (feature weights)
coef_img = decoder.coef_img_

# Create a mean functional image for background
mean_func = mean_img(img)

# Plot the discriminative map
fig = plt.figure(figsize=(12, 4))
plot_stat_map(
    coef_img, 
    bg_img=mean_func,
    title='Multivariate pattern: Listening vs Rest',
    display_mode='z',
    cut_coords=5,
    colorbar=True,
    cmap='RdBu_r'  # Red-blue colormap
)
show()
```

The resulting map shows which voxels have positive weights (contributing to "listening" classification) and negative weights (contributing to "rest" classification).

### Searchlight analysis

For more localized analysis, we can use a searchlight approach that runs the classification within small spheres throughout the brain:

```python
from nilearn.decoding import SearchLight
from nilearn.plotting import plot_glass_brain

# Create a searchlight decoder
searchlight = SearchLight(
    mask_img=None,             # Will create automatic mask
    radius=4.0,                # 4mm radius spheres
    estimator='svc',           # Support Vector Classifier
    cv=3,                      # 3-fold cross-validation for speed
    scoring='accuracy',        # Accuracy metric
    n_jobs=1                   # Number of parallel jobs
)

print("Running searchlight analysis (this may take a few minutes)...")
searchlight.fit(img, labels)

# Plot the searchlight accuracy map
fig = plt.figure(figsize=(10, 6))
plot_glass_brain(
    searchlight.scores_,
    colorbar=True,
    title='Searchlight Classification Accuracy',
    plot_abs=False,
    vmax=1.0,
    vmin=0.3,
    cmap='hot'
)
show()
```

The searchlight map reveals which brain regions contain the most informative patterns for distinguishing between experimental conditions.

### Feature selection and interpretation

We can investigate which features (voxels) are most important for classification:

```python
# Get feature importance from the decoder
feature_scores = decoder.cv_scores_

# Create a mask of the most discriminative voxels
from nilearn.image import threshold_img
from nilearn.masking import apply_mask, unmask

# Threshold the coefficient map to show only strong weights
thresh_coef_img = threshold_img(coef_img, threshold='95%')

# Extract coordinates of discriminative voxels
from nilearn.plotting import find_parcellation_cut_coords
coords = find_parcellation_cut_coords(thresh_coef_img)

print("Most discriminative regions (MNI coordinates):")
for i, coord in enumerate(coords[:10]):  # Show top 10
    print(f"Region {i+1}: ({coord[0]:3.0f}, {coord[1]:3.0f}, {coord[2]:3.0f})")
```

### Time series analysis of discriminative patterns

We can also examine how the discriminative pattern evolves over time:

```python
# Extract the average signal from discriminative regions
from nilearn.maskers import NiftiMasker

# Create a mask from the thresholded coefficient map
masker = NiftiMasker(
    mask_img=thresh_coef_img,
    standardize=True,
    detrend=True
)

# Extract time series from discriminative voxels
discriminative_signal = masker.fit_transform(img)
mean_signal = np.mean(discriminative_signal, axis=1)

# Plot the time series with condition labels
plt.figure(figsize=(12, 6))
plt.plot(mean_signal, 'b-', linewidth=2, label='Discriminative pattern')

# Highlight listening periods
for i in range(n_scans):
    if labels[i] == 'listening':
        plt.axvspan(i-0.4, i+0.4, alpha=0.3, color='red')

plt.xlabel('Time (TRs)')
plt.ylabel('Signal intensity (z-score)')
plt.title('Time series of discriminative multivariate pattern')
plt.legend()
plt.grid(True, alpha=0.3)
show()
```

## Comparison with univariate analysis

Let's compare the multivariate results with traditional univariate analysis:

```python
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img

# Univariate GLM analysis
glm = FirstLevelModel(
    t_r=7.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=0.01
)

# Create events DataFrame for GLM
events_df = pd.DataFrame({
    'trial_type': ['listening'] * (n_scans // 2),
    'onset': np.arange(7, n_scans*7, 14),  # Every other block
    'duration': [7] * (n_scans // 2)       # 7 second blocks
})

glm.fit(img, events_df)
z_map = glm.compute_contrast('listening', output_type='z_score')

# Threshold the univariate map
thresh_z_map, threshold = threshold_stats_img(
    z_map, alpha=0.001, height_control='fpr'
)

# Compare multivariate and univariate results side by side
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Multivariate pattern
plot_stat_map(
    coef_img,
    bg_img=mean_func,
    axes=axes[0],
    title='Multivariate pattern (MVPA)',
    display_mode='z',
    cut_coords=3,
    colorbar=True,
    cmap='RdBu_r'
)

# Univariate activation
plot_stat_map(
    thresh_z_map,
    bg_img=mean_func,
    axes=axes[1],
    title='Univariate activation (GLM)',
    display_mode='z',
    cut_coords=3,
    colorbar=True,
    cmap='hot'
)

plt.tight_layout()
show()
```

## Advanced multivariate techniques

### Cross-classification (generalization)

We can test whether patterns learned from one condition generalize to another:

```python
# Split data into two runs for cross-classification
n_half = n_scans // 2
run1_img = image.index_img(img, slice(0, n_half))
run2_img = image.index_img(img, slice(n_half, n_scans))
run1_labels = labels[:n_half]
run2_labels = labels[n_half:]

# Train on run 1, test on run 2
decoder_cross = Decoder(
    estimator='svc',
    mask_strategy='background',
    standardize=True,
    screening_percentile=20
)

decoder_cross.fit(run1_img, run1_labels)
predictions = decoder_cross.predict(run2_img)
cross_accuracy = accuracy_score(run2_labels, predictions)

print(f"Cross-run generalization accuracy: {cross_accuracy:.3f}")
print(f"Within-run accuracy: {mean_accuracy:.3f}")
```

### Temporal generalization

We can examine how patterns evolve over time by training and testing at different time points:

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Create a time-resolved analysis
time_decoder = []
for lag in range(-3, 4):  # Test different temporal offsets
    if lag >= 0:
        train_img = image.index_img(img, slice(0, n_scans-lag))
        test_labels = labels[lag:]
        train_labels = labels[:-lag] if lag > 0 else labels
    else:
        train_img = image.index_img(img, slice(-lag, n_scans))
        test_labels = labels[:lag]
        train_labels = labels[-lag:]
    
    # Quick decoder for temporal analysis
    masker_temp = NiftiMasker(standardize=True)
    X = masker_temp.fit_transform(train_img)
    
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf, X, train_labels, cv=3)
    time_decoder.append(np.mean(scores))

# Plot temporal generalization
plt.figure(figsize=(10, 6))
lags = range(-3, 4)
plt.plot(lags, time_decoder, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
plt.xlabel('Temporal lag (TRs)')
plt.ylabel('Classification accuracy')
plt.title('Temporal generalization of multivariate patterns')
plt.legend()
plt.grid(True, alpha=0.3)
show()
```

## Summary

Multivariate analysis offers several advantages over univariate approaches:

1. **Sensitivity**: Can detect distributed patterns that might be missed by voxel-wise analysis
2. **Prediction**: Enables decoding of cognitive states and mental representations
3. **Robustness**: Less sensitive to precise anatomical alignment across subjects
4. **Information**: Provides insights into the information content of brain regions

Key considerations:

- **Overfitting**: Requires careful cross-validation due to high dimensionality
- **Interpretation**: Results are less directly interpretable than univariate maps
- **Computational cost**: More computationally intensive than univariate methods
- **Statistical inference**: Different statistical frameworks than traditional fMRI analysis

Nilearn provides powerful tools for multivariate analysis that integrate well with the broader Python scientific ecosystem, making it easier to apply sophisticated machine learning techniques to neuroimaging data.