# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:00:22 2025

@author: ywan3672
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#%%
# Choose your model type: 'logistic', 'random_forest', or 'svc'
model_type = 'random_forest'  # <<< change this line to switch models

# norm_hyp_ratio = [0.75,0.25]
# norm_hyp_ratio = [0.25,0.75]
norm_hyp_ratio = [0.7,0.3]
# norm_hyp_ratio = [0.5,0.5]

#%% Generate synthetic radiomics-like dataset (imbalanced)
X, y = make_classification(
    n_samples=1000,
    n_features=20,  # 100 total features
    n_informative=6, # Only 5 are actually informative
    n_redundant=4, # 5 are linear combinations of the informative ones
    n_classes=2,
    class_sep = 0.5, # Controls class separation (lower = more overlap = harder, default is 1 = moderate difficulty, 0.5-0.8 can be used to simulate real-world heterogeneity)
    flip_y = 0.1, # You can increase this to simulate label noise. Default 0.01, incrase to 0.05 to give mild label noise.
    weights=norm_hyp_ratio,  # 25% hypoxic
    random_state=42 
)

#Add Gaussian noise to simulate measurement variability
noise_std = 0.1  # standard deviation of the Gaussian noise
X += np.random.normal(loc=0.0, scale=noise_std, size=X.shape)
#%% Select and make model
if model_type == 'logistic':
    model = LogisticRegression(solver='liblinear')
elif model_type == 'random_forest':
    model = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
elif model_type == 'svc':
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    )
else:
    raise ValueError("Invalid model_type. Choose 'logistic', 'random_forest', or 'svc'.")

#%% Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=5,
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

#%% Plot
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Score", marker='o')
plt.plot(train_sizes, test_mean, label="Validation Score", marker='s')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.title(f"Learning Curve ({model_type.replace('_', ' ').title()}) - ROC AUC")
plt.xlabel("Training Set Size")
plt.ylabel("Score (ROC AUC)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
