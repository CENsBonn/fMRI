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

# Decoding

In the previous section, we used
{doc}`univariate analysis<analysis-univariate>`
to answer the question:
"What happens in the brain when a participant is exposed to an auditory stimulus?".
In this chapter, we will reverse the question by asking: "Given a pattern of
brain activity, is the participant being exposed to an auditory stimulus?".
The latter approach is commonly known as *decoding*.
Nilearn is able to perform decoding using Multi-Voxel Pattern Analysis (MVPA).

If your goal is to build an explanatory model of brain activity in order to
learn something new about how stimuli affect brain activity, decoding is arguably
[not the right approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC5797513/).
However, if your goal is to build a predictive model without necessarily
understanding the underlying mechanisms, decoding is a powerful method.

## Prediction accuracy

We will consider a basic binary classification task based on the
{doc}`SPM dataset<analysis-data>`:
predicting whether the participant was listening to the auditory stimulus or at
rest, using only the brain activity patterns.
The following code uses the Nilearn `Decoder` class to train a Support Vector
Classifier in order to predict the experimental condition from brain activity
patterns. The prediction accuracy is calculated based on 5-fold
cross-validation:

```python
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_spm_auditory
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker
from nilearn.decoding import Decoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

subject_data = fetch_spm_auditory()
fmri_img = load_img(subject_data.func)
events = pd.read_table(subject_data.events)

n_scans = fmri_img.shape[-1]
tr = 7.0

labels = ['rest'] * n_scans

for _, event in events.iterrows():
    onset_tr = int(event['onset'] / tr)
    duration_tr = int(event['duration'] / tr)
    for t in range(onset_tr, onset_tr + duration_tr):
        if t < n_scans:
            labels[t] = 'listening'

labels = np.array(labels)

decoder = Decoder(
    estimator='svc',          # Support Vector Classifier
    mask_strategy='epi',      # Use EPI mask (better for functional data)
    standardize=True,         # Standardize features across voxels
    screening_percentile=10,  # Use top 10% most variable voxels
    cv=5,                     # 5-fold cross-validation
    scoring='accuracy',       # Classification accuracy
    n_jobs=1                  # Single thread for stability
)

decoder.fit(fmri_img, labels)

cv_scores_dict = decoder.cv_scores_
cv_scores = list(cv_scores_dict.values())[0]

mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Decoding Results:")
print(f"Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
print(f"Individual fold scores: {[float(f"{s:.3f}") for s in cv_scores]}")
print(f"Chance level (balanced): 0.500")

from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(cv_scores, 0.5)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.6f}")
```

Output:

```
Decoding Results:
Cross-validation accuracy: 0.810 ± 0.057
Individual fold scores: [0.824, 0.824, 0.882, 0.706, 0.812]
Chance level (balanced): 0.500
T-statistic: 10.795
P-value: 0.000418
```

The output above shows that our model is able to predict the correct label
(listening vs rest) with an 81% accuracy.
This is significantly above the 50% chance level we would expect
if the model was randomly guessing.
The t-test confirms this is statistically
significant (p < 0.001), indicating that brain activity patterns reliably
distinguish between listening and rest conditions. The relatively small
standard deviation (5.7%) suggests consistent performance across
cross-validation folds.

## ROC curve

We can also evaluate the classifier performance using ROC (Receiver Operating
Characteristic) analysis and calculate the Area Under the Curve (AUC):

```python
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_spm_auditory
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

subject_data = fetch_spm_auditory()
fmri_img = load_img(subject_data.func)
events = pd.read_table(subject_data.events)

n_scans = fmri_img.shape[-1]
tr = 7.0

labels = ['rest'] * n_scans

for _, event in events.iterrows():
    onset_tr = int(event['onset'] / tr)
    duration_tr = int(event['duration'] / tr)
    for t in range(onset_tr, onset_tr + duration_tr):
        if t < n_scans:
            labels[t] = 'listening'

labels = np.array(labels)

masker = NiftiMasker(
    mask_strategy='epi',
    standardize=True,
    screening_percentile=10
)

brain_data = masker.fit_transform(fmri_img)

scaler = StandardScaler()
brain_data_scaled = scaler.fit_transform(brain_data)

y_binary = (labels == 'listening').astype(int)

clf_roc = SVC(kernel='linear', C=1.0, probability=True, random_state=42)

print("Computing cross-validated ROC curve...")
y_proba_cv = cross_val_predict(clf_roc, brain_data_scaled, y_binary, cv=5, method='predict_proba')
y_scores = y_proba_cv[:, 1]  # Probability of positive class (listening)

fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
roc_auc = auc(fpr, tpr)

print(f"Cross-validated AUC: {roc_auc:.3f}")

fig = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for MVPA Decoding: Listening vs Rest')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
```

Output:
```
Computing cross-validated ROC curve...
Cross-validated AUC: 0.785
```

```{figure} ./images/decoding_roc_curve.png
:name: decoding-roc-curve
:width: 70%

ROC curve for MVPA decoding showing the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity). The AUC of 0.785 indicates good discriminative ability, substantially above the 0.5 chance level (diagonal dashed line). This demonstrates that the classifier can reliably distinguish between listening and rest states from brain activity patterns.
```

The AUC of 0.785 indicates good discriminative performance, substantially above
the 0.5 chance level. This means the classifier can distinguish between
listening and rest states with high reliability, with the ROC curve in
{numref}`decoding-roc-curve`
showing strong separation from the diagonal chance line.
