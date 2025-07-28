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

In the previous section on {doc}`univariate analysis<analysis-univariate>`, we analyzed individual voxels in isolation using the General Linear Model. Each voxel's time series was fitted independently, testing for activation one voxel at a time. This *univariate* approach is powerful for identifying localized brain activations.

Multi-Voxel Pattern Analysis (MVPA) takes a fundamentally different approach by examining patterns of activity across multiple voxels simultaneously. Instead of asking "*which* voxels are active?", MVPA asks "*what information* can we decode from patterns of brain activity?"

## What is MVPA?

MVPA uses machine learning techniques to classify or predict cognitive states, experimental conditions, or behavioral outcomes based on distributed patterns of brain activity. Key characteristics include:

- **Pattern-based**: Analyzes spatial patterns across voxels rather than individual voxel activations
- **Predictive**: Uses supervised learning to decode information from brain activity
- **Sensitive**: Can detect distributed information that univariate methods might miss
- **Flexible**: Applicable to classification, regression, and representational analysis

## MVPA with Nilearn

Nilearn provides excellent tools for MVPA through its `decoding` module. Let's start with a classic example using the Haxby dataset, which contains visual object recognition data perfect for demonstrating classification.

### Face vs House Classification

We'll train a classifier to distinguish between viewing faces and houses based on patterns of brain activity:

```python
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.decoding import Decoder
from nilearn.image import index_img
from scipy.stats import ttest_1samp

# Load the Haxby dataset
haxby_dataset = datasets.fetch_haxby()

# Load functional data and behavioral labels for first subject
fmri_filename = haxby_dataset.func[0]
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=' ')

print(f"Available conditions: {behavioral.labels.unique()}")

# Select face vs house conditions for binary classification
conditions = ['face', 'house']
condition_mask = behavioral['labels'].isin(conditions)

# Filter the data
fmri_img = index_img(fmri_filename, condition_mask)
labels = behavioral['labels'][condition_mask]

print(f"Selected {len(labels)} samples")
print(f"Face samples: {sum(labels == 'face')}")
print(f"House samples: {sum(labels == 'house')}")
```

Output:
```
Available conditions: ['rest' 'scissors' 'face' 'cat' 'shoe' 'house' 'scrambledpix' 'bottle' 'chair']
Selected 216 samples
Face samples: 108
House samples: 108
```

Now let's create and train our MVPA decoder:

```python
# Create a decoder with optimized parameters
decoder = Decoder(
    estimator='svc',          # Support Vector Classifier
    mask_strategy='epi',      # Use EPI mask (better for functional data)
    standardize=True,         # Standardize features
    screening_percentile=5,   # Use top 5% most variable voxels
    cv=5,                     # 5-fold cross-validation
    scoring='accuracy'        # Accuracy metric
)

# Fit the decoder
print("Training classifier...")
decoder.fit(fmri_img, labels)

# Extract cross-validation scores
cv_scores_dict = decoder.cv_scores_
# Both classes have identical scores in binary classification
cv_scores = list(cv_scores_dict.values())[0]

mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
print(f"Individual fold scores: {cv_scores}")
print(f"Chance level: 0.500")

# Statistical significance test
t_stat, p_value = ttest_1samp(cv_scores, 0.5)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.6f}")
```

Output:
```
Training classifier...
Cross-validation accuracy: 0.950 ± 0.056
Individual fold scores: [0.841, 0.977, 0.977, 0.953, 1.000]
Chance level: 0.500
T-statistic: 15.974
P-value: 0.000090
```

Excellent! The classifier achieves 95% accuracy, demonstrating that face and house viewing conditions produce highly distinguishable patterns of brain activity.

### Visualizing Discriminative Patterns

A key advantage of MVPA is that we can examine which brain regions contribute most to successful classification:

```python
from nilearn.plotting import plot_stat_map, show
from nilearn.image import mean_img
import matplotlib.pyplot as plt

# Get the discriminative map (coefficient weights)
# For binary classification, we extract weights for one class
coef_img_dict = decoder.coef_img_
coef_img = coef_img_dict['face']  # Weights favoring face classification

# Create a mean functional image as background
mean_func = mean_img(fmri_img)

# Plot the discriminative pattern
fig = plt.figure(figsize=(12, 4))
plot_stat_map(
    coef_img,
    bg_img=mean_func,
    title='MVPA Discriminative Pattern: Face vs House',
    display_mode='z',
    cut_coords=5,
    colorbar=True,
    cmap='RdBu_r',  # Red-blue colormap
    threshold=0.01   # Show only strong weights
)
show()
```

The discriminative map reveals which voxels contribute to face vs house classification:
- **Red regions**: Voxels with positive weights that favor "face" classification
- **Blue regions**: Voxels with negative weights that favor "house" classification
- **Strong patterns** typically appear in visual cortex, especially the fusiform face area for faces and parahippocampal place area for houses

### Searchlight Analysis

Searchlight analysis runs classification within small spheres across the brain, revealing which local regions contain the most discriminative information:

```python
from nilearn.decoding import SearchLight
from nilearn.plotting import plot_glass_brain

# Create a searchlight decoder
# Note: This is computationally intensive, so we use a subset for demonstration
searchlight = SearchLight(
    mask_img=None,          # Automatic brain mask
    radius=4.0,             # 4mm radius spheres
    estimator='svc',        # Support Vector Classifier
    cv=3,                   # 3-fold cross-validation
    scoring='accuracy',     # Classification accuracy
    n_jobs=1               # Single thread for stability
)

# For demonstration, use a subset of the data to reduce computation time
subset_img = index_img(fmri_img, slice(0, 60))  # First 60 volumes
subset_labels = labels.iloc[:60]

print("Running searchlight analysis...")
print("(Using subset of data for demonstration - full analysis takes longer)")
searchlight.fit(subset_img, subset_labels)

# Plot the searchlight accuracy map
fig = plt.figure(figsize=(12, 8))
plot_glass_brain(
    searchlight.scores_,
    colorbar=True,
    title='Searchlight MVPA: Local Classification Accuracy',
    threshold=0.5,          # Show only above-chance regions
    vmax=1.0,
    vmin=0.5,
    cmap='hot'
)
show()
```

The searchlight map highlights brain regions where local patterns contain sufficient information for classification. Bright regions indicate areas with high local decoding accuracy.

### Understanding Feature Importance

We can investigate which voxels contribute most to successful classification:

```python
from nilearn.image import threshold_img
from nilearn.plotting import find_peaks

# Threshold the coefficient map to identify the most discriminative voxels
thresh_coef_img = threshold_img(coef_img, threshold='95%')

# Find peak coordinates of discriminative regions
peaks = find_peaks(thresh_coef_img, min_distance=20, threshold=0.01)

print("Most discriminative regions (MNI coordinates):")
for i, (x, y, z, intensity) in enumerate(peaks[:10]):
    print(f"Peak {i+1}: ({x:3.0f}, {y:3.0f}, {z:3.0f}) - Weight: {intensity:.3f}")

# Visualize the thresholded discriminative map
fig = plt.figure(figsize=(12, 4))
plot_stat_map(
    thresh_coef_img,
    bg_img=mean_func,
    title='Top 5% Most Discriminative Voxels',
    display_mode='z',
    cut_coords=5,
    colorbar=True,
    cmap='RdBu_r'
)
show()
```

This analysis identifies the specific brain regions most critical for face vs house discrimination, typically including visual cortex areas specialized for these categories.

### Time Series Analysis of Discriminative Patterns

We can examine how discriminative patterns evolve over time by extracting signals from the most informative voxels:

```python
from nilearn.maskers import NiftiMasker

# Create a masker using the thresholded discriminative map
masker = NiftiMasker(
    mask_img=thresh_coef_img,
    standardize=True,
    detrend=True
)

# Extract time series from discriminative voxels
discriminative_signal = masker.fit_transform(fmri_img)
mean_signal = np.mean(discriminative_signal, axis=1)

# Create time series plot with condition labels
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the discriminative pattern signal
ax.plot(mean_signal, 'b-', linewidth=2, label='Discriminative pattern strength')

# Highlight different conditions with colors
face_indices = np.where(labels == 'face')[0]
house_indices = np.where(labels == 'house')[0]

for idx in face_indices:
    ax.axvspan(idx-0.4, idx+0.4, alpha=0.3, color='red', label='Face' if idx == face_indices[0] else "")
for idx in house_indices:
    ax.axvspan(idx-0.4, idx+0.4, alpha=0.3, color='blue', label='House' if idx == house_indices[0] else "")

ax.set_xlabel('Time (TRs)')
ax.set_ylabel('Signal strength (z-score)')
ax.set_title('Time Course of Face vs House Discriminative Pattern')
ax.legend()
ax.grid(True, alpha=0.3)
show()
```

This visualization shows how the strength of the discriminative pattern varies over time, with clear differences between face and house viewing periods.

## MVPA vs Univariate Analysis

Let's compare MVPA results with traditional univariate analysis to understand their complementary strengths:

```python
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img

# For univariate analysis, we need to create a proper events DataFrame
# Create events based on our labels
events_list = []
current_condition = None
onset_time = 0

for i, condition in enumerate(labels):
    if condition != current_condition:
        if current_condition is not None:
            # End the previous event
            duration = i * 2.5 - onset_time  # Assuming 2.5s TR
            events_list.append({
                'trial_type': current_condition,
                'onset': onset_time,
                'duration': duration
            })
        # Start new event
        current_condition = condition
        onset_time = i * 2.5

# Add the final event
if current_condition is not None:
    duration = len(labels) * 2.5 - onset_time
    events_list.append({
        'trial_type': current_condition,
        'onset': onset_time,
        'duration': duration
    })

events_df = pd.DataFrame(events_list)
print(f"Created {len(events_df)} events for GLM analysis")

# Fit GLM model
glm = FirstLevelModel(
    t_r=2.5,                # TR for Haxby dataset
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=0.01
)

glm.fit(fmri_img, events_df)

# Compute contrast: face - house
z_map = glm.compute_contrast('face - house', output_type='z_score')

# Threshold the univariate map
thresh_z_map, threshold = threshold_stats_img(
    z_map, alpha=0.001, height_control='fpr'
)

print(f"Univariate analysis threshold: z > {threshold:.2f}")
```

### Visual Comparison

```python
# Compare MVPA and univariate results
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# MVPA discriminative pattern
plot_stat_map(
    coef_img,
    bg_img=mean_func,
    axes=axes[0],
    title='MVPA: Discriminative Pattern (Face vs House)',
    display_mode='z',
    cut_coords=6,
    colorbar=True,
    cmap='RdBu_r'
)

# Univariate activation map
plot_stat_map(
    thresh_z_map,
    bg_img=mean_func,
    axes=axes[1],
    title='Univariate GLM: Face - House Contrast',
    display_mode='z',
    cut_coords=6,
    colorbar=True,
    cmap='hot',
    threshold=threshold
)

plt.tight_layout()
show()
```

### Key Differences

**MVPA (Top panel):**
- Shows **patterns** of voxels that collectively discriminate conditions
- Red/blue regions indicate voxels with positive/negative classification weights
- Reveals distributed information across brain regions
- Focuses on **predictive** patterns rather than average differences

**Univariate GLM (Bottom panel):**
- Shows regions with **significant average differences** between conditions
- Identifies voxels that are consistently more active for one condition
- Tests statistical significance at each voxel independently
- Focuses on **localized activations** rather than distributed patterns

## Advanced MVPA Techniques

### Cross-Classification (Generalization)

A crucial test of MVPA is whether patterns learned from one dataset generalize to independent data:

```python
from sklearn.metrics import accuracy_score, classification_report

# Split data into two halves for cross-validation
n_half = len(labels) // 2

# First half for training
train_img = index_img(fmri_img, slice(0, n_half))
train_labels = labels.iloc[:n_half]

# Second half for testing
test_img = index_img(fmri_img, slice(n_half, len(labels)))
test_labels = labels.iloc[n_half:]

print(f"Training samples: {len(train_labels)}")
print(f"Testing samples: {len(test_labels)}")

# Create decoder for cross-classification
cross_decoder = Decoder(
    estimator='svc',
    mask_strategy='epi',
    standardize=True,
    screening_percentile=5,
    cv=None  # No cross-validation, we're doing manual train/test split
)

# Train on first half
print("Training on first half of data...")
cross_decoder.fit(train_img, train_labels)

# Test on second half
print("Testing on second half of data...")
predictions = cross_decoder.predict(test_img)
generalization_accuracy = accuracy_score(test_labels, predictions)

print(f"\nGeneralization Results:")
print(f"Cross-run accuracy: {generalization_accuracy:.3f}")
print(f"Within-run accuracy: {mean_accuracy:.3f}")
print(f"\nClassification Report:")
print(classification_report(test_labels, predictions))
```

Output:
```
Training samples: 108
Testing samples: 108
Training on first half of data...
Testing on second half of data...

Generalization Results:
Cross-run accuracy: 0.898
Within-run accuracy: 0.950

Classification Report:
              precision    recall  f1-score   support

        face       0.89      0.91      0.90        54
       house       0.91      0.89      0.90        54

    accuracy                           0.90       108
   macro avg       0.90      0.90      0.90       108
weighted avg       0.90      0.90      0.90       108
```

High generalization accuracy (89.8%) demonstrates that the learned patterns are robust and not overfitted to the training data.

### Multi-Class Classification

MVPA can handle more complex classification problems beyond binary decisions:

```python
# Extend to multi-class classification: face, house, and cat
multi_conditions = ['face', 'house', 'cat']
multi_condition_mask = behavioral['labels'].isin(multi_conditions)

# Filter data for multi-class problem
multi_fmri_img = index_img(haxby_dataset.func[0], multi_condition_mask)
multi_labels = behavioral['labels'][multi_condition_mask]

print(f"Multi-class samples: {len(multi_labels)}")
for condition in multi_conditions:
    count = sum(multi_labels == condition)
    print(f"{condition.capitalize()}: {count} samples")

# Multi-class decoder
multi_decoder = Decoder(
    estimator='svc',
    mask_strategy='epi',
    standardize=True,
    screening_percentile=5,
    cv=5,
    scoring='accuracy'
)

print("\nTraining multi-class classifier...")
multi_decoder.fit(multi_fmri_img, multi_labels)

# Extract scores for multi-class classification
multi_cv_scores = list(multi_decoder.cv_scores_.values())[0]
multi_mean_accuracy = np.mean(multi_cv_scores)

print(f"Multi-class accuracy: {multi_mean_accuracy:.3f} ± {np.std(multi_cv_scores):.3f}")
print(f"Chance level (3 classes): {1/3:.3f}")

# Statistical test
t_stat, p_value = ttest_1samp(multi_cv_scores, 1/3)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.6f}")
```

Output:
```
Multi-class samples: 324
Face: 108 samples
House: 108 samples
Cat: 108 samples

Training multi-class classifier...
Multi-class accuracy: 0.901 ± 0.048
Chance level (3 classes): 0.333
T-statistic: 26.398
P-value: 0.000008
```

The multi-class classifier achieves 90% accuracy, well above the 33% chance level, demonstrating MVPA's ability to distinguish between multiple cognitive states.

## When to Use MVPA

### MVPA is Ideal For:

1. **Decoding cognitive states**: "What is the participant thinking about?"
2. **Classification tasks**: Distinguishing between experimental conditions
3. **Information mapping**: Finding where specific information is represented
4. **Distributed patterns**: Detecting information spread across regions
5. **Prediction**: Forecasting behavior or clinical outcomes

### MVPA vs Univariate: Complementary Approaches

| Aspect | MVPA | Univariate GLM |
|--------|------|----------------|
| **Question** | "What can we decode?" | "Where is activation?" |
| **Sensitivity** | Distributed patterns | Localized activation |
| **Statistics** | Cross-validation accuracy | Statistical significance |
| **Interpretation** | Predictive patterns | Average differences |
| **Multiple comparisons** | Inherent control | Requires correction |

## Best Practices for MVPA

### 1. Cross-Validation Strategy
```python
# Always use proper cross-validation
decoder = Decoder(
    cv=5,  # At minimum 5-fold CV
    scoring='accuracy',  # Choose appropriate metric
    estimator='svc'      # Linear SVM often works well
)
```

### 2. Feature Selection
```python
# Use feature selection to reduce overfitting
decoder = Decoder(
    screening_percentile=5,  # Select top 5% most variable voxels
    standardize=True         # Always standardize features
)
```

### 3. Proper Statistical Testing
```python
# Test against chance level, not zero
from scipy.stats import ttest_1samp
n_classes = len(np.unique(labels))
chance_level = 1.0 / n_classes
t_stat, p_value = ttest_1samp(cv_scores, chance_level)
```

### 4. Validate Generalization
```python
# Test on independent data when possible
train_decoder.fit(training_data, training_labels)
test_accuracy = train_decoder.score(test_data, test_labels)
```

## Summary

MVPA revolutionizes neuroimaging analysis by:

- **Revealing distributed information** that univariate methods miss
- **Enabling prediction** of cognitive states and behaviors  
- **Providing interpretable patterns** through classification weights
- **Offering robust statistics** through cross-validation frameworks

With Nilearn's powerful MVPA tools, researchers can decode the information content of brain activity, moving beyond traditional "where is it active?" to "what information is encoded?" This paradigm shift opens new possibilities for understanding brain function and developing brain-based applications.

### Key Takeaways

1. MVPA focuses on **patterns** rather than individual voxel activations
2. **Cross-validation** is essential for avoiding overfitting
3. **Feature selection** improves performance and interpretability
4. **Generalization testing** validates real-world applicability
5. MVPA and univariate analysis are **complementary**, not competing approaches

MVPA represents a powerful addition to the neuroimaging toolkit, enabling researchers to ask and answer fundamentally new questions about brain function and mental representation.
