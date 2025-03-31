# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:53:05 2025

@author: ywan3672
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%% 🔧 Choose your model type: 'logistic', 'random_forest', or 'svc'
model_type = 'logistic'  # <<< Change this value to switch models
sample_sizes = [30, 50, 100, 160, 240, 320, 1000]
n_repeats = 50

# norm_hyp_ratio = [0.75,0.25]
# norm_hyp_ratio = [0.25,0.75]
# norm_hyp_ratio = [0.7,0.3]
norm_hyp_ratio = [0.3,0.7]
# norm_hyp_ratio = [0.5,0.5]

#%% 1. Generate synthetic radiomics-like data
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=6,
    n_redundant=4,
    n_classes=2,
    weights=norm_hyp_ratio,
    class_sep=0.7,
    flip_y=0.05,
    random_state=42
)
#Add Gaussian noise to simulate measurement variability
noise_std = 0.1  # standard deviation of the Gaussian noise
X += np.random.normal(loc=0.0, scale=noise_std, size=X.shape)
#%% 2. Define the model
if model_type == 'logistic':
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
elif model_type == 'random_forest':
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
elif model_type == 'svc':
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    )
else:
    raise ValueError("Invalid model type. Choose 'logistic', 'random_forest', or 'svc'.")

# 3. Simulation parameters
auc_results = {size: [] for size in sample_sizes}

#%% 4. Simulate AUC variability
from sklearn.model_selection import StratifiedKFold

for size in sample_sizes:
    for _ in range(n_repeats):
        X_sample, y_sample = resample(X, y, n_samples=size, stratify=y, random_state=None)
        auc = cross_val_score(model, X_sample, y_sample,
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                              scoring='roc_auc').mean()
        auc_results[size].append(auc)

#%% 5. Plot the variability
plt.figure(figsize=(10, 6))
plt.boxplot([auc_results[size] for size in sample_sizes], labels=sample_sizes)
plt.title(f"AUC Variability by Sample Size ({model_type.replace('_', ' ').title()})")
plt.xlabel("Sample Size")
plt.ylabel("Cross-Validated AUC")
plt.grid(True)
plt.tight_layout()
plt.show()
