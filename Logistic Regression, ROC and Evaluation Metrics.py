
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# Create dataset
X = np.array([
    [2, 3],
    [1, 2],
    [3, 1],
    [6, 5],
    [7, 8],
    [8, 6]
])

y = np.array([0, 0, 0, 1, 1, 1])

# Train Logistic Regression model
model = LogisticRegression().fit(X, y)

# Predict classes
y_pred = model.predict(X)

# Get predicted probabilities for class 1
probs = model.predict_proba(X)[:, 1]

# Display evaluation metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1:", f1_score(y, y_pred))
print("AUC:", roc_auc_score(y, probs))

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y, probs)

# Plot ROC curve
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "--", label="Random Classifier")

plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
